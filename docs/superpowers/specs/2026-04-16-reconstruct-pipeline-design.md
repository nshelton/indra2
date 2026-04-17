# Half-res Reconstruct Pipeline — Design

**Date:** 2026-04-16
**Status:** approved (pending implementation)

## Goal

Integrate the half-res raymarch trace into the full-res frame via a dedicated reconstruction pass, using a joint-bilateral (depth-aware) spatial filter. The design also establishes a TAA-ready scaffold: history ping-pong textures and bindings stay in place, with temporal accumulation deferred to a follow-up.

## Motivation

`present.metal` currently fakes the upsample with a nearest half-res fetch. This produces blocky output and no edge preservation. A proper reconstruction pass:

- Resolves the half→full upsample with edge-aware weights, avoiding bleed across silhouettes and depth discontinuities.
- Separates concerns: **reconstruct** resolves the image, **present** finishes it (tonemap, exposure — later).
- Puts the existing jitter (`raymarch.metal:87`) and history ping-pong textures (`main.cpp:60-61`) on a path to temporal accumulation without requiring topology changes when TAA lands.

Film grain / noise is explicitly **not** synthesized as a post-process — jittered undersampling + reconstruction produces natural artifacts.

## Pipeline

```
raymarch_kernel    (half-res)   →  writes current_color, current_depth
reconstruct_kernel (full-res)   →  reads current_color, current_depth, history_in
                                   writes history_out (= this frame's resolved color)
present_kernel     (full-res)   →  reads history_out, writes output
blit_to_screen(output)
```

## Texture layout

No new allocations — the existing textures in `main.cpp` fit the new topology:

| Texture | Res | Format | Role |
|---|---|---|---|
| `tex_current_color` | half | RGBA16F | raymarch output (hit color) |
| `tex_current_depth` | half | R32F | raymarch output (distance `t`) |
| `tex_history_a` / `tex_history_b` | full | RGBA16F | ping-pong; reconstruct writes current slot, reads previous slot |
| `tex_output` | full | RGBA16F | present writes, blitted to screen |

**Key invariant:** "output of reconstruct" and "history for next frame" are the same texture. Spatial-only and TAA share topology — when TAA lands, the same `history_out` write becomes the temporally-integrated result without texture reshuffling.

First frame: `history_in` is zero-init. Acceptable because the spatial-only kernel ignores `history_in` entirely.

## `reconstruct.metal`

### Bindings

```metal
kernel void reconstruct_kernel(
    texture2d<float, access::read>    current_color [[texture(0)]],  // half-res
    texture2d<float, access::read>    current_depth [[texture(1)]],  // half-res
    texture2d<float, access::read>    history_in    [[texture(2)]],  // full-res, unused in spatial-only
    texture2d<float, access::write>   history_out   [[texture(3)]],  // full-res, this frame's resolved color
    constant FrameUniforms&           frame         [[buffer(0)]],
    uint2 gid                                         [[thread_position_in_grid]]);
```

### `@param` controls

Parsed by `shader_manager` from the shader header, exposed via ImGui:

```
// @param filter_sigma       float 0.1 3.0 1.0
// @param filter_depth_sigma float 0.001 0.5 0.05
// @param edge_aware         float 0.0 1.0 1.0
```

`edge_aware` is a float 0–1 (not a bool) so it fits the existing `@param` float-4 pipeline with no special casing. 0 = pure gaussian, 1 = full joint-bilateral, interpolated in between.

### Algorithm (per full-res pixel `gid`)

1. Map to half-res sample coord: `half_coord = (float2(gid) + 0.5) * 0.5 - 0.5`
2. Reference depth: `z_ref = current_depth.read(round(half_coord))`
3. Loop 3×3 half-res neighborhood around `floor(half_coord)`:
   - `off = (tap_coord + 0.5) - (half_coord + 0.5)`
   - `w_spatial = exp(-0.5 * dot(off, off) / (sigma * sigma))`
   - `z = current_depth.read(tap_coord)`
   - `w_depth   = mix(1.0, exp(-abs(z - z_ref) / (z_ref * depth_sigma)), edge_aware)`
   - `sum  += color * w_spatial * w_depth`
   - `wsum += w_spatial * w_depth`
4. `resolved = sum / max(wsum, 1e-6)`
5. **TAA seam** (commented no-op today):
   ```metal
   // float4 history = history_in.read(reproject(gid));
   // resolved = mix(history, resolved, alpha);
   ```
6. `history_out.write(resolved, gid)`

Taps outside the texture are clamped (`clamp` on coords before `.read`).

## `present.metal`

Becomes a near-passthrough stub. Bindings narrow to just what it needs — no depth, no history:

```metal
kernel void present_kernel(
    texture2d<float, access::read>   in_color [[texture(0)]],  // full-res, from reconstruct
    texture2d<float, access::write>  output   [[texture(1)]],  // full-res
    constant FrameUniforms&          frame    [[buffer(0)]],
    uint2 gid                                   [[thread_position_in_grid]])
{
    if (gid.x >= output.get_width() || gid.y >= output.get_height()) return;
    float4 color = in_color.read(gid);
    // TODO: tonemap / exposure / vignette go here
    output.write(color, gid);
}
```

No synthesized film grain. Tonemapping/exposure/vignette added later as needed.

## `main.cpp` changes

1. Register the new shader after the existing `register_shader` calls:
   ```cpp
   shaders.register_shader("reconstruct.metal", "reconstruct_kernel");
   ```

2. Replace the 2-pass render block (`main.cpp:174-202`) with three dispatches:
   ```cpp
   // Pass 1: raymarch (half-res) — unchanged
   backend.dispatch({
       .pipeline_id = rm_pipeline,
       .grid_width = half_w, .grid_height = half_h,
       .textures = {tex_current_color, tex_current_depth},
       .buffers  = {buf_uniforms}
   });

   // Pass 2: reconstruct (full-res)
   int history_read  = ping ? tex_history_a : tex_history_b;
   int history_write = ping ? tex_history_b : tex_history_a;
   backend.dispatch({
       .pipeline_id = reconstruct_pipeline,
       .grid_width = w, .grid_height = h,
       .textures = {tex_current_color, tex_current_depth, history_read, history_write},
       .buffers  = {buf_uniforms}
   });

   // Pass 3: present (full-res)
   backend.dispatch({
       .pipeline_id = present_pipeline,
       .grid_width = w, .grid_height = h,
       .textures = {history_write, tex_output},
       .buffers  = {buf_uniforms}
   });

   ping = !ping;
   backend.blit_to_screen(tex_output);
   ```

3. ImGui: add `reconstruct.metal` params to the existing `render_shader_params` panel so bilateral sliders appear next to raymarch's.

4. No texture allocations change. No resize handler changes needed — existing code already resizes all five textures.

## Out of scope (future work)

- **Temporal accumulation.** The `reproject()` helper, neighborhood clamping (variance or min-max), disocclusion masks, and the `mix(history, resolved, alpha)` line. The TAA seam in `reconstruct.metal` is where that lands.
- **Tonemapping / exposure / vignette.** Goes in `present.metal` where the `TODO` marker sits.
- **Full-res depth output.** Not needed until TAA reprojection arrives; will be added then (nearest-depth upsample from half-res).
- **Performance tuning.** Threadgroup sizing, texture sampler vs `.read()`, 2×2 vs 3×3 window, any caching. Defer until measured.

## Success criteria

- `reconstruct.metal` compiles and loads through the existing shader hot-reload path.
- Output at full-res is visibly smoother than the current nearest upsample, with preserved silhouettes at the fractal's edges and at the grid plane.
- `filter_sigma`, `filter_depth_sigma`, `edge_aware` sliders appear in the ImGui panel and visibly change the image.
- `edge_aware = 0` produces a pure-gaussian result (bleeds edges); `edge_aware = 1` preserves them.
- `present.metal` is a passthrough; disabling reconstruct (by replacing its write with zeros) blacks the screen, confirming reconstruct is load-bearing and present is not masking its output.
- No regression in `Show Grid` toggle, camera controls, or shader hot-reload.
