# TAA (Temporal Anti-Aliasing) Design

**Date:** 2026-04-16
**Status:** approved (pending implementation)
**Builds on:** `2026-04-16-reconstruct-pipeline-design.md`

## Goal

Fill in the TAA seam scaffolded in `reconstruct.metal:48-51` with a minimal, robust temporal accumulation. The spatial-only reconstruct currently flickers under sub-pixel jitter because each frame is an independent half-res snapshot. Blending with a reprojected history resolves the jitter offsets across frames and delivers the super-resolution that jitter is paying for.

## Motivation

Observed: with the spatial joint-bilateral in place, the output still flickers frame-to-frame because the raymarch's per-frame Halton jitter shifts the entire sample grid by up to a half-pixel. Spatial filtering averages neighbors within one frame but cannot cancel cross-frame variation.

TAA is the dual: for each full-res pixel, find where that world-space point sat in the previous frame's buffer, sample the history color, clamp it to the current frame's local neighborhood (to reject disocclusions and feature changes), and blend. Over a handful of frames this integrates the jittered samples into a stable, higher-effective-resolution image.

## Scope (minimal)

- In-place fill of the TAA seam in `reconstruct.metal`. **No new pass, no new shader file.**
- Camera-motion-only reprojection (sufficient because the scene has no moving geometry).
- AABB (min/max over 3×3 current-frame taps) neighborhood clamping with a user-tunable padding scale.
- Single fixed `taa_alpha` blend slider. Not velocity-adaptive.
- One new full-res depth texture written by reconstruct. Used in-kernel for reprojection; no ping-pong.

## Out of scope (future iterations)

- Motion vectors / velocity buffer from raymarch (not needed: static geometry).
- Catmull-Rom / bicubic history sampling.
- Variance-based neighborhood clipping.
- Motion-adaptive blend alpha.
- Full-res depth ping-pong / history depth buffer (required only if we add depth-based disocclusion rejection later).
- Per-pixel confidence / rejection masks surfaced to ImGui.

## Data flow

```
raymarch (half-res)  →  current_color, current_depth
reconstruct (full-res):
    reads  current_color, current_depth, history_in (prev frame's output)
    writes history_out (this frame's resolved color), reconstructed_depth (full-res)
present (full-res passthrough)  →  output
blit_to_screen
```

The ping-pong of `tex_history_a/b` is unchanged — reconstruct writes to `history_write` (the "this frame" slot), which becomes `history_in` next frame. The new `tex_reconstructed_depth` is written and then read inside the same kernel only; no cross-pass, no ping-pong.

## Textures

Existing 5-texture set in `main.cpp` gains **one new full-res texture**:

| Texture | Res | Format | New? | Role |
|---|---|---|---|---|
| `tex_current_color` | half | RGBA16F | existing | raymarch color output |
| `tex_current_depth` | half | R32F | existing | raymarch depth output (world-space `t`) |
| `tex_history_a` / `tex_history_b` | full | RGBA16F | existing | ping-pong — one is `history_in`, other is `history_out` |
| `tex_output` | full | RGBA16F | existing | present writes, blitted to screen |
| **`tex_reconstructed_depth`** | **full** | **R32F** | **new** | **dominant-tap depth, consumed by in-kernel reprojection** |

Resize handler adds one line for the new texture.

## `reconstruct.metal` kernel

### Bindings (grow 4 → 5)

```metal
kernel void reconstruct_kernel(
    texture2d<float, access::read>    current_color         [[texture(0)]],  // half-res
    texture2d<float, access::read>    current_depth         [[texture(1)]],  // half-res
    texture2d<float, access::read>    history_in            [[texture(2)]],  // full-res
    texture2d<float, access::write>   history_out           [[texture(3)]],  // full-res
    texture2d<float, access::write>   reconstructed_depth   [[texture(4)]],  // full-res (new)
    constant FrameUniforms&           frame                 [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]);
```

### `@param` controls (grow 3 → 5)

Existing:
```
// @param filter_sigma       float 0.1 3.0 1.0
// @param filter_depth_sigma float 0.001 0.5 0.05
// @param edge_aware         float 0.0 1.0 1.0
```

New:
```
// @param taa_alpha          float 0.0 1.0 0.1    // 1.0 = TAA off; 0.1 = strong smoothing
// @param taa_clamp_scale    float 0.1 3.0 1.0    // AABB padding; larger = more ghosting through
```

These sit in `frame.recon_params[3..4].x`. The existing `@param` packing loop already handles up to 8 slots.

### Algorithm

For each full-res pixel `gid`:

**Phase A — Spatial bilateral (existing + two new trackers):**

Same 3×3 loop as today, additionally tracking:
- `ngb_min`, `ngb_max` (`float4` each): min/max of the 9 tap colors — the clamp AABB.
- `best_w` (float), `dom_depth` (float): the weight and depth of the tap with highest `w = w_s * w_d`.

After the loop:
```
resolved_current = sum / max(wsum, 1e-6)
```

**Phase B — Reprojection (skipped for frame 0, sky/miss, off-screen, behind camera):**

```
alpha = taa_alpha
history_clamped = resolved_current   // fallback when reproject is skipped

if (frame.frame_index > 0 && dom_depth < max_dist_threshold):
    // Build world-space hit point using the SAME camera basis as raymarch
    ndc = (float2(gid) + 0.5) * frame.inv_resolution * 2 - 1
    ndc.y = -ndc.y
    aspect = frame.resolution.x * frame.inv_resolution.y
    hf = tan(frame.camera_fov * 0.5)
    ray_dir = normalize(camera_fwd + camera_right * ndc.x * hf * aspect
                                   + camera_up    * ndc.y * hf)
    world = camera_pos + ray_dir * dom_depth

    prev_clip = frame.prev_view_proj * float4(world, 1.0)

    if (prev_clip.w <= 0):
        alpha = 1.0
    else:
        prev_ndc = prev_clip.xy / prev_clip.w
        prev_uv  = prev_ndc * 0.5 + 0.5
        prev_uv.y = 1.0 - prev_uv.y   // match pixel-from-top convention

        if (any(prev_uv < 0) || any(prev_uv > 1)):
            alpha = 1.0
        else:
            // Manual bilinear fetch from history_in
            prev_pix = prev_uv * float2(full_res_wh)
            i0 = int2(floor(prev_pix - 0.5))
            f  = prev_pix - 0.5 - float2(i0)
            c00/c10/c01/c11 = clamped neighbors
            history = bilinear(c00, c10, c01, c11, f)

            // AABB clamp with padding
            center = 0.5 * (ngb_min + ngb_max)
            half_e = 0.5 * (ngb_max - ngb_min) * taa_clamp_scale
            history_clamped = clamp(history, center - half_e, center + half_e)
else:
    alpha = 1.0
```

`max_dist_threshold` mirrors the raymarch's hardcoded `max_dist = 100.0` (`raymarch.metal:97`). Hardcode `dom_depth < 99.0` directly in the reconstruct kernel — matches the existing style of shader-local constants. If `max_dist` ever gets promoted to a uniform, both sites update together. Sky/miss pixels get `alpha = 1.0` because there's no meaningful world-space point to reproject — you'd end up sampling history along a ray at infinity, which is noise.

**Phase C — Blend and write:**

```
final = mix(history_clamped, resolved_current, alpha)
history_out.write(final, gid)
reconstructed_depth.write(float4(dom_depth, 0, 0, 0), gid)
```

### Edge cases — all collapse to `alpha = 1.0` (TAA off for that pixel)

| Case | Why |
|---|---|
| `frame.frame_index == 0` | `history_in` is an uninitialized private Metal texture — its contents are undefined (flagged in Task 4 code review of the reconstruct pipeline). |
| `dom_depth >= max_dist_threshold` | Sky/miss pixel; no world-space point to reproject. |
| `prev_clip.w <= 0` | World point was behind prev-frame camera. |
| `prev_uv` outside `[0,1]²` | History tap is off-screen in prev frame; disocclusion from camera motion. |

Current-pixel-only under these conditions is the correct behavior: it degrades gracefully to the spatial-only result you have today.

### Algorithm correctness notes

- **Ray construction must match raymarch exactly.** If reconstruct builds a different ray from the same `(pixel, depth)` pair, the computed `world_pos` won't be on the surface raymarch traced, and reprojection will ghost. The inline code above copies the camera-ray math verbatim from `make_camera_ray` in `common.metal`, omitting only the half-res `+ frame.jitter` offset (reconstruct is full-res and not jittered).
- **Depth is world-space `t`**, not NDC z — do not feed it through `inv_view_proj`. `world = camera_pos + ray_dir * t` is the direct way.
- **`prev_view_proj` is a standard perspective matrix** computed by `Camera::get_view_proj` (`src/gui.cpp:36-41`) using the same FOV and aspect as raymarch's manual ray build. They agree within floating-point error on where a world point lands in NDC.
- **Y-axis convention: verified Y-up NDC.** Raymarch's manual ray build uses Y-up NDC (pixel y=0 → ndc.y=+1 via `ndc.y = -ndc.y`). `mat4::perspective` in `src/math_util.h:40-48` is the standard OpenGL-style perspective: `out[5] = 1/t` with `clip.w = -view.z`, so a point above the camera (view.y > 0, view.z < 0) gives ndc.y = +1 (top). Conventions agree. The reprojection formula `prev_uv = (prev_clip.xy / prev_clip.w) * 0.5 + 0.5; prev_uv.y = 1.0 - prev_uv.y` is correct as written.

## `main.cpp` changes

Five-line delta:

1. Allocate the new texture (next to existing `create_texture` calls):
   ```cpp
   int tex_reconstructed_depth = backend.create_texture({w, h, TextureDesc::R32Float, true, "reconstructed_depth"});
   ```

2. Resize handler appends:
   ```cpp
   backend.resize_texture(tex_reconstructed_depth, w, h);
   ```

3. Reconstruct dispatch's `.textures` gains the new slot at the end:
   ```cpp
   .textures = {tex_current_color, tex_current_depth, history_read, history_write, tex_reconstructed_depth},
   ```

4. `@param` packing loop — **no change required**. Already caps at `i < 8` and handles any parsed count ≤ 8.

5. ImGui panel — **no change required**. The existing `render_shader_params(shaders.get_params_mut("reconstruct.metal"))` picks up the two new sliders automatically via the `@param` system.

## Success criteria

- With the camera at rest and `taa_alpha` default (0.1), the jitter flicker visibly converges to a stable image over ~5–10 frames. (It's stable, not frozen — moving the camera re-breaks convergence, as expected.)
- Pan the camera slowly: the image stays sharp without noticeable ghosting on silhouettes.
- Pan quickly or rotate suddenly: some transient ghosting on thin features is acceptable for minimal TAA. Clamping should prevent long trails.
- `taa_alpha = 1.0` disables TAA entirely and produces exactly the current spatial-only output — a clean "off" switch.
- `taa_clamp_scale` slider changes ghosting characteristics visibly: small values (0.1–0.5) tighten clamping (more rejection, less smoothing), large values (2–3) loosen it (more ghost-through).
- Edge cases produce no black or NaN pixels: sky regions stay stable; first frame is not black (falls back to spatial-only via `alpha = 1.0`); reprojection off-screen does not crash or produce visual garbage.
- No regression: `filter_sigma`, `filter_depth_sigma`, `edge_aware`, `Show Grid`, camera controls, shader hot-reload all continue to work as in the reconstruct pipeline.

## Risks / known limitations

- **First-frame fallback uses `frame.frame_index == 0`.** After a window resize, `tex_history_a/b` get reallocated as new `MTLStorageModePrivate` textures with undefined contents, but `frame_index` is not reset. The clamp will drag garbage into the current neighborhood range, so output won't be broken — just a few frames of muted convergence. Tolerable for minimal; revisit if visible.
- **Sub-pixel reprojection accuracy depends on `dom_depth` alignment with color.** We chose option (b) from Q2 (dominant-weight tap's depth) specifically to minimize this at silhouettes. If ghosting is observed at fractal edges, the first knob to check is `taa_clamp_scale`; if that doesn't resolve it, the depth source is suspect.
- **Bilinear history sampling smears slightly.** Over many frames of an accumulated still camera, this would monotonically blur. The per-frame `alpha = 0.1` blend brings in fresh samples, which counteracts this — but if over-smoothing becomes visible at rest, Catmull-Rom sampling is the first upgrade.
- **No velocity buffer means moving lights / animated uniforms are treated as disocclusions** by the clamp. This is fine today (static scene, static lighting) but would need revisiting if time-varying shaders are added.
