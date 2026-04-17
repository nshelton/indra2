# Reconstruct Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fake half-res upsample in `present.metal` with a dedicated reconstruction pass that performs a joint-bilateral upsample from half-res to full-res, with a TAA-ready scaffold in place.

**Architecture:** Three-pass pipeline — raymarch (half-res) writes color + depth, reconstruct (full-res) performs depth-aware 3×3 upsample and writes to ping-ponged history textures, present (full-res) becomes a passthrough stub for future tonemapping. Filter parameters are exposed through the existing `@param`/ImGui system via a new `recon_params` block in `FrameUniforms`.

**Tech Stack:** C++20, Metal Shading Language, SDL3, CMake, ImGui.

**Spec:** `docs/superpowers/specs/2026-04-16-reconstruct-pipeline-design.md`

## Notes for the implementer

- **No unit tests exist in this project** — verification is manual: build, run the app, look at the output, toggle sliders. Each task ends with a build-and-run check.
- **Metal shader hot-reload is wired in** (`shader_manager.cpp:41-51`). Changes to `.metal` files take effect without a rebuild; compile errors surface in the ImGui error panel (`gui.cpp:254-262`). You can leave the app running while iterating on shader code.
- **The `FrameUniforms` struct is mirrored in two places:** `src/types.h` (C++) and `shaders/common.metal` (Metal). Changes must be made in both and keep the same layout — the existing `_pad` fields in the C++ struct match `packed_float3` + `float` alignment on the Metal side. When in doubt, run the app and watch for garbage uniforms.
- **Build command:** `cmake --build build -j` from the repo root. If `build/` doesn't exist, run `cmake -B build` first.
- **Run command:** `./build/fractal-engine` from the repo root.
- **Commits** are listed in each task but should only be executed when you (the human) approve. The planner/executor will pause for authorization.

## File structure

| Path | Status | Responsibility |
|---|---|---|
| `src/types.h` | modify | Add `recon_params` block to `FrameUniforms` |
| `shaders/common.metal` | modify | Mirror `recon_params` in Metal struct |
| `shaders/reconstruct.metal` | rewrite | Joint-bilateral upsample kernel with `@param` controls and TAA seam |
| `shaders/present.metal` | rewrite | Narrow to passthrough stub; drop depth/history bindings |
| `src/main.cpp` | modify | Register reconstruct shader, copy its params into uniforms, restructure dispatches, add ImGui panel section |

---

### Task 1: Add `recon_params` block to `FrameUniforms`

The current `FrameUniforms.params[32][4]` is filled only from `raymarch.metal`'s parsed `@param` list (`main.cpp:163-167`). A dedicated block for reconstruct avoids magic offsets and keeps shader boundaries clean. Eight `float4` slots is plenty (spec uses three).

**Files:**
- Modify: `src/types.h:55-57`
- Modify: `shaders/common.metal:30-33`

- [ ] **Step 1: Extend `FrameUniforms` in `types.h`**

Replace the tail of the struct (`src/types.h:54-58`):

```cpp
    // Shader params: each param occupies one float4 regardless of actual size.
    float params[32][4];           // raymarch.metal params
    float recon_params[8][4];      // reconstruct.metal params
    uint32_t param_count;
    uint32_t recon_param_count;
    uint32_t _pad5[2];
};
```

- [ ] **Step 2: Mirror the change in `common.metal`**

Replace the tail of the Metal struct (`shaders/common.metal:30-33`):

```metal
    float4 params[32];          // raymarch.metal params
    float4 recon_params[8];     // reconstruct.metal params
    uint   param_count;
    uint   recon_param_count;
    uint2  _pad5;
};
```

- [ ] **Step 3: Build**

Run: `cmake --build build -j`
Expected: Build succeeds. No warnings about struct size mismatch (there shouldn't be — the compiler doesn't cross-check C++ and Metal).

- [ ] **Step 4: Run and sanity-check**

Run: `./build/fractal-engine`
Expected: App launches, fractal renders as before (no change in output). Close with ESC.

This verifies the struct extension didn't break alignment of existing fields. If the fractal looks garbled or the camera is broken, the layout is off — check padding before moving on.

- [ ] **Step 5: Commit**

```bash
git add src/types.h shaders/common.metal
git commit -m "add recon_params block to FrameUniforms for reconstruct shader"
```

---

### Task 2: Rewrite `reconstruct.metal` with joint-bilateral kernel

Replace the placeholder kernel (`shaders/reconstruct.metal:1-23`) with the full joint-bilateral upsample. Bindings are kept compatible with the dispatch shape that will be wired up in Task 4. The shader compiles on hot-reload — you should see "Compiled reconstruct.metal successfully (3 params)" in the log once it's registered in Task 3.

**Files:**
- Rewrite: `shaders/reconstruct.metal`

- [ ] **Step 1: Replace the file contents**

Replace the entire contents of `shaders/reconstruct.metal`:

```metal
// @param filter_sigma       float 0.1 3.0 1.0
// @param filter_depth_sigma float 0.001 0.5 0.05
// @param edge_aware         float 0.0 1.0 1.0

kernel void reconstruct_kernel(
    texture2d<float, access::read>    current_color [[texture(0)]],  // half-res
    texture2d<float, access::read>    current_depth [[texture(1)]],  // half-res
    texture2d<float, access::read>    history_in    [[texture(2)]],  // full-res, unused in spatial-only
    texture2d<float, access::write>   history_out   [[texture(3)]],  // full-res
    constant FrameUniforms&           frame         [[buffer(0)]],
    uint2                             gid           [[thread_position_in_grid]]
) {
    uint2 full_res = uint2(history_out.get_width(), history_out.get_height());
    if (gid.x >= full_res.x || gid.y >= full_res.y) return;

    float sigma       = frame.recon_params[0].x;
    float depth_sigma = frame.recon_params[1].x;
    float edge_aware  = frame.recon_params[2].x;

    int2  half_res  = int2(current_color.get_width(), current_color.get_height());
    float2 half_coord = (float2(gid) + 0.5) * 0.5 - 0.5;
    int2   base       = int2(floor(half_coord));

    int2  ref_tap = clamp(int2(round(half_coord)), int2(0), half_res - 1);
    float z_ref   = current_depth.read(uint2(ref_tap)).r;

    float4 sum  = float4(0.0);
    float  wsum = 0.0;

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int2  tap    = clamp(base + int2(dx, dy), int2(0), half_res - 1);
            float2 off   = float2(tap) - half_coord;
            float  w_s   = exp(-0.5 * dot(off, off) / (sigma * sigma));

            float  z     = current_depth.read(uint2(tap)).r;
            float  z_rel = (z_ref > 1e-4) ? (abs(z - z_ref) / (z_ref * depth_sigma)) : 0.0;
            float  w_d   = mix(1.0, exp(-z_rel), edge_aware);

            float  w     = w_s * w_d;
            sum  += current_color.read(uint2(tap)) * w;
            wsum += w;
        }
    }

    float4 resolved = sum / max(wsum, 1e-6);

    // TAA seam — no-op today. When temporal accumulation lands:
    //   float2 prev_uv = reproject(gid, frame);
    //   float4 history = history_in.read(clamp_uv(prev_uv));
    //   resolved = mix(history, resolved, alpha);

    history_out.write(resolved, gid);
}
```

- [ ] **Step 2: Build (no dispatch yet, so should still succeed)**

Run: `cmake --build build -j`
Expected: Build succeeds. The shader isn't compiled at build time — Metal compilation happens at runtime when `register_shader` is called in Task 3.

- [ ] **Step 3: Commit**

```bash
git add shaders/reconstruct.metal
git commit -m "add joint-bilateral upsample kernel to reconstruct.metal"
```

---

### Task 3: Register reconstruct shader and wire its params

Register the shader, pack its `@param` values into `recon_params`, and surface its sliders in the ImGui panel. At the end of this task the shader is loaded but not yet dispatched — output is unchanged (still going through the old `present.metal`).

**Files:**
- Modify: `src/main.cpp:49-51` (add `register_shader`)
- Modify: `src/main.cpp:162-167` (extend uniform packing)
- Modify: `src/main.cpp:234-236` (extend ImGui panel)

- [ ] **Step 1: Register the shader**

Change `main.cpp:49-51` from:

```cpp
    ShaderManager shaders(backend, shader_dir);
    shaders.register_shader("raymarch.metal", "raymarch_kernel");
    shaders.register_shader("present.metal", "present_kernel");
```

to:

```cpp
    ShaderManager shaders(backend, shader_dir);
    shaders.register_shader("raymarch.metal", "raymarch_kernel");
    shaders.register_shader("reconstruct.metal", "reconstruct_kernel");
    shaders.register_shader("present.metal", "present_kernel");
```

- [ ] **Step 2: Extend uniform packing**

Change `main.cpp:162-167` from:

```cpp
        // Pack shader params
        const auto& params = shaders.get_params("raymarch.metal");
        uniforms.param_count = (uint32_t)params.size();
        for (int i = 0; i < (int)params.size() && i < 32; i++) {
            std::memcpy(uniforms.params[i], params[i].current_val, sizeof(float) * 4);
        }
```

to:

```cpp
        // Pack raymarch params
        const auto& rm_params = shaders.get_params("raymarch.metal");
        uniforms.param_count = (uint32_t)rm_params.size();
        for (int i = 0; i < (int)rm_params.size() && i < 32; i++) {
            std::memcpy(uniforms.params[i], rm_params[i].current_val, sizeof(float) * 4);
        }

        // Pack reconstruct params
        const auto& rc_params = shaders.get_params("reconstruct.metal");
        uniforms.recon_param_count = (uint32_t)rc_params.size();
        for (int i = 0; i < (int)rc_params.size() && i < 8; i++) {
            std::memcpy(uniforms.recon_params[i], rc_params[i].current_val, sizeof(float) * 4);
        }
```

- [ ] **Step 3: Add ImGui section for reconstruct params**

Change `main.cpp:234-236` from:

```cpp
        if (ImGui::CollapsingHeader("Shader Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
            render_shader_params(shaders.get_params_mut("raymarch.metal"));
        }
```

to:

```cpp
        if (ImGui::CollapsingHeader("Shader Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
            render_shader_params(shaders.get_params_mut("raymarch.metal"));
        }
        if (ImGui::CollapsingHeader("Reconstruction", ImGuiTreeNodeFlags_DefaultOpen)) {
            render_shader_params(shaders.get_params_mut("reconstruct.metal"));
        }
```

- [ ] **Step 4: Build**

Run: `cmake --build build -j`
Expected: Build succeeds.

- [ ] **Step 5: Run and verify shader loads and sliders appear**

Run: `./build/fractal-engine`
Expected:
- Terminal log shows `[shader] Compiled reconstruct.metal successfully (3 params)`.
- ImGui window has a new "Reconstruction" panel with three sliders: `filter_sigma`, `filter_depth_sigma`, `edge_aware`.
- Moving the sliders has NO visible effect yet — reconstruct isn't dispatched.
- The fractal still renders exactly as before (blocky half-res nearest upsample via old present).

If the sliders don't appear, check the log for `Error in reconstruct.metal` — most likely a Metal compile error will name the issue.

Close with ESC.

- [ ] **Step 6: Commit**

```bash
git add src/main.cpp
git commit -m "register reconstruct shader and wire its params through uniforms/ImGui"
```

---

### Task 4: Restructure dispatches + simplify `present.metal`

This is the atomic switchover. `present.metal` is narrowed to a passthrough stub and the render loop in `main.cpp` is restructured to three passes. Do both in the same commit — they're not independently correct.

**Files:**
- Rewrite: `shaders/present.metal`
- Modify: `src/main.cpp:174-205` (render loop dispatch block + blit)

- [ ] **Step 1: Rewrite `present.metal` as a passthrough stub**

Replace the entire contents of `shaders/present.metal`:

```metal
kernel void present_kernel(
    texture2d<float, access::read>   in_color [[texture(0)]],  // full-res, from reconstruct
    texture2d<float, access::write>  output   [[texture(1)]],  // full-res
    constant FrameUniforms&          frame    [[buffer(0)]],
    uint2                            gid      [[thread_position_in_grid]]
) {
    uint2 full_res = uint2(output.get_width(), output.get_height());
    if (gid.x >= full_res.x || gid.y >= full_res.y) return;

    float4 color = in_color.read(gid);

    // TODO: tonemap / exposure / vignette go here

    output.write(color, gid);
}
```

- [ ] **Step 2: Restructure the render loop in `main.cpp`**

Replace `main.cpp:174-205` (the block starting with `// Pass 1: Raymarch (half-res)` and ending after `backend.blit_to_screen(tex_output);`) with:

```cpp
        // Pass 1: Raymarch (half-res)
        int rm_pipeline = shaders.get_pipeline("raymarch_kernel");
        if (rm_pipeline >= 0) {
            backend.dispatch({
                .pipeline_id = rm_pipeline,
                .grid_width = half_w,
                .grid_height = half_h,
                .threadgroup_w = 16,
                .threadgroup_h = 16,
                .textures = {tex_current_color, tex_current_depth},
                .buffers = {buf_uniforms}
            });
        }

        // Pass 2: Reconstruct (full-res, half → full with joint-bilateral)
        int rc_pipeline = shaders.get_pipeline("reconstruct_kernel");
        int history_read  = ping ? tex_history_a : tex_history_b;
        int history_write = ping ? tex_history_b : tex_history_a;
        if (rc_pipeline >= 0) {
            backend.dispatch({
                .pipeline_id = rc_pipeline,
                .grid_width = w,
                .grid_height = h,
                .threadgroup_w = 16,
                .threadgroup_h = 16,
                .textures = {tex_current_color, tex_current_depth, history_read, history_write},
                .buffers = {buf_uniforms}
            });
        }

        // Pass 3: Present (full-res passthrough stub)
        int present_pipeline = shaders.get_pipeline("present_kernel");
        if (present_pipeline >= 0) {
            backend.dispatch({
                .pipeline_id = present_pipeline,
                .grid_width = w,
                .grid_height = h,
                .threadgroup_w = 16,
                .threadgroup_h = 16,
                .textures = {history_write, tex_output},
                .buffers = {buf_uniforms}
            });
        }

        ping = !ping;

        // Blit to screen
        backend.blit_to_screen(tex_output);
```

- [ ] **Step 3: Build**

Run: `cmake --build build -j`
Expected: Build succeeds.

- [ ] **Step 4: Run and visually verify the pipeline is live**

Run: `./build/fractal-engine`
Expected:
- Fractal renders at full-res; edges look smoother than the previous blocky upsample.
- Terminal log shows both `reconstruct.metal` and `present.metal` compiled successfully.
- No errors in the ImGui shader error panel.

If the screen is black: reconstruct probably failed to compile — check the terminal + error panel. If the screen shows garbage: likely a uniform layout mismatch from Task 1 — recheck `_pad5` alignment.

Close with ESC.

- [ ] **Step 5: Commit**

```bash
git add shaders/present.metal src/main.cpp
git commit -m "wire reconstruct pass into render loop and narrow present to passthrough"
```

---

### Task 5: Visual verification against spec success criteria

No code changes — this task confirms the pipeline behaves as the spec demands. If any check fails, open a follow-up debugging task rather than silently tweaking.

- [ ] **Step 1: Run the app**

Run: `./build/fractal-engine`

- [ ] **Step 2: `edge_aware = 0` produces a pure gaussian (bleeds across edges)**

In the "Reconstruction" ImGui panel, set `edge_aware` slider to 0.0. Observe: silhouettes of the fractal against the background should soften/bleed visibly compared to `edge_aware = 1`. Half-tone halos around sharp features are expected.

- [ ] **Step 3: `edge_aware = 1` preserves edges**

Set `edge_aware` to 1.0. Observe: silhouettes should be crisper; no halo around depth discontinuities.

- [ ] **Step 4: `filter_sigma` visibly changes smoothness**

With `edge_aware = 1.0`, sweep `filter_sigma` from 0.1 → 3.0. Observe: image goes from nearly-nearest (blocky-ish, preserves jitter noise) to blurrier.

- [ ] **Step 5: `filter_depth_sigma` controls edge tightness**

With `edge_aware = 1.0` and `filter_sigma ≈ 1.5`, sweep `filter_depth_sigma`. Small values (~0.005) → very sharp but potentially noisy; large values (~0.3) → behaves closer to pure gaussian.

- [ ] **Step 6: Existing features still work**

- Toggle "Show Grid" checkbox → grid appears/disappears on the XZ plane.
- Rotate/pan/zoom with the mouse → camera responds normally.
- Edit `shaders/reconstruct.metal` in another editor (e.g. change `sigma * sigma` to `sigma * sigma * 2.0`) and save → the app reloads the shader live, image updates. Revert the change.

- [ ] **Step 7: Check performance baseline**

Observe the `%.1f fps (%.2f ms)` readout in the ImGui window. Record the number — no strict target, but it should be comparable to before (reconstruct adds one full-res pass, so a slight dip is expected; a massive dip suggests threadgroup config is off or a loop is running way too large).

- [ ] **Step 8: No commit needed — this was verification only**

---

## Self-review against spec

**Spec coverage:**
- ✅ Three-pass pipeline (raymarch → reconstruct → present) — Tasks 2, 4.
- ✅ Texture layout (no new allocations; reuses history ping-pong as reconstruct output) — Task 4.
- ✅ `reconstruct.metal` bindings, `@param` controls, joint-bilateral algorithm, TAA seam — Task 2.
- ✅ `present.metal` narrowed to passthrough stub, no synthesized grain — Task 4.
- ✅ `main.cpp` registers reconstruct, wires params, restructures dispatches, adds ImGui panel — Tasks 3, 4.
- ✅ Success criteria (edge_aware toggle, sigma/depth_sigma sliders, grid toggle unaffected, hot-reload working) — Task 5.
- ✅ **Gap found and filled:** spec didn't specify how reconstruct's `@param` values reach the GPU. Task 1 adds a `recon_params` block to `FrameUniforms` and Task 3 wires the copy.

**Type consistency check:**
- `FrameUniforms.recon_params[8][4]` (C++) matches `float4 recon_params[8]` (Metal): both 128 bytes.
- Trailing layout: C++ has `uint32_t param_count; uint32_t recon_param_count; uint32_t _pad5[2];` (16 bytes). Metal mirrors with `uint param_count; uint recon_param_count; uint2 _pad5;` (16 bytes). ✅
- Texture binding indices match between reconstruct.metal (Task 2) and the dispatch call (Task 4): `current_color=0, current_depth=1, history_in=2, history_out=3`. ✅
- Present texture indices match: `in_color=0, output=1`. ✅
