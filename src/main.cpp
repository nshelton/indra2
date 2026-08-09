#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "metal_backend.h"
#include "shader_manager.h"
#include "gui.h"
#include "math_util.h"
#include "state_serializer.h"
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>
#include <iostream>
#include <unistd.h>

int main(int argc, char* argv[]) {
    // 1. SDL Init
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        std::cerr << "SDL_Init failed: " << SDL_GetError() << "\n";
        return 1;
    }

    SDL_Window* window = SDL_CreateWindow("fractal-engine",
        1920, 1080,
        SDL_WINDOW_METAL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
    if (!window) {
        std::cerr << "SDL_CreateWindow failed: " << SDL_GetError() << "\n";
        return 1;
    }

    // 2. Metal backend init
    MetalBackend backend;
    if (!backend.init(window)) {
        std::cerr << "Metal backend init failed\n";
        return 1;
    }

    // 3. ImGui init
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplSDL3_InitForMetal(window);
    backend.imgui_init();

    // 4. Shader manager init — resolve shader dir relative to executable
    std::string shader_dir = SDL_GetBasePath();
    shader_dir += "shaders";
    std::cout << "[init] Shader directory: " << shader_dir << "\n";

    ShaderManager shaders(backend, shader_dir);
    shaders.register_shader("raymarch.metal", "raymarch_kernel");
    shaders.register_shader("pathtrace.metal", "pathtrace_kernel");
    shaders.register_shader("reconstruct.metal", "reconstruct_kernel");
    shaders.register_shader("present.metal", "present_kernel");

    // 5. Create textures
    uint32_t w = backend.drawable_width();
    uint32_t h = backend.drawable_height();
    uint32_t half_w = w / 2, half_h = h / 2;

    int tex_current_color = backend.create_texture({half_w, half_h, TextureDesc::RGBA16Float, true, "current_color"});
    int tex_current_depth = backend.create_texture({half_w, half_h, TextureDesc::R32Float, true, "current_depth"});
    int tex_output        = backend.create_texture({w, h, TextureDesc::RGBA16Float, true, "output"});
    // History is RGBA32Float: with accumulation alpha = 1/N, half-float
    // precision stalls convergence after a few hundred frames.
    int tex_history_a     = backend.create_texture({w, h, TextureDesc::RGBA32Float, true, "history_a"});
    int tex_history_b     = backend.create_texture({w, h, TextureDesc::RGBA32Float, true, "history_b"});
    // Reconstructed depth ping-pongs like history: reconstruct reads last
    // frame's for the TAA disocclusion test while writing this frame's.
    int tex_recon_depth_a = backend.create_texture({w, h, TextureDesc::R32Float, true, "recon_depth_a"});
    int tex_recon_depth_b = backend.create_texture({w, h, TextureDesc::R32Float, true, "recon_depth_b"});
    bool ping = false;

    // 6. Uniform buffers — a ring of 3, since up to 3 frames can be in flight
    // on the GPU and a single shared buffer would let the CPU overwrite
    // uniforms mid-read (torn matrices, off-by-a-frame reprojection).
    int buf_uniforms[3];
    for (int i = 0; i < 3; i++) {
        buf_uniforms[i] = backend.create_buffer(sizeof(FrameUniforms), "uniforms" + std::to_string(i));
    }

    // Depth readback: center patch of the half-res depth texture, read one frame later
    constexpr uint32_t DEPTH_PATCH = 16;
    int buf_depth_read = backend.create_buffer(DEPTH_PATCH * DEPTH_PATCH * sizeof(float), "depth_readback");

    // Pick readback: depth patch under the cursor, for double-click trackball centering
    constexpr uint32_t PICK_PATCH = 4;
    int buf_pick_read = backend.create_buffer(PICK_PATCH * PICK_PATCH * sizeof(float), "pick_readback");

    // 7. Camera + renderer mode
    Camera camera;
    int renderer_mode = 1;  // 0 = raymarch, 1 = path trace

    // 8. State persistence — register every persisted value once; the
    // serializer handles save/load/change-detection for all of them.
    std::string state_path = std::string(SDL_GetBasePath()) + "state.json";
    StateSerializer state(state_path);
    state.enum_field("/camera/mode", (int*)&camera.mode, {"trackball", "fps"});
    state.float3_field("/camera/pos", camera.pos);
    state.float3_field("/camera/target", camera.target);
    state.float_field("/camera/fov", &camera.fov);
    state.bool_field("/camera/show_grid", &camera.show_grid);
    state.float_field("/camera/rotate_speed", &camera.rotate_speed);
    state.float_field("/camera/pan_speed", &camera.pan_speed);
    state.float_field("/camera/zoom_speed", &camera.zoom_speed);
    state.float_field("/camera/keyboard_speed", &camera.keyboard_speed);
    state.bool_field("/camera/adaptive_speed", &camera.adaptive_speed);
    state.enum_field("/renderer", &renderer_mode, {"raymarch", "pathtrace"});
    state.load(shaders);
    camera.zoom_speed = std::clamp(camera.zoom_speed, 0.01f, 1.0f);
    camera.keyboard_speed = std::clamp(camera.keyboard_speed, 0.01f, 2.0f);

    // 9. Timing
    uint64_t start_time = SDL_GetPerformanceCounter();
    uint64_t freq = SDL_GetPerformanceFrequency();
    uint32_t frame_index = 0;
    float prev_time = 0;
    float prev_vp[16];
    mat4::identity(prev_vp);

    // 10. Accumulation state
    int prev_renderer_mode = renderer_mode;
    uint32_t accum_frames = 0;
    uint32_t frames_since_move = 0;
    FrameUniforms prev_uniforms = {};

    std::string exe_path = std::string(SDL_GetBasePath()) + "fractal-engine";

    // Main loop
    bool running = true;
    bool restart = false;
    while (running) {
        // --- Events ---
        SDL_Event event;
        std::vector<SDL_Event> events;
        bool pick_requested = false;
        float pick_x = 0, pick_y = 0;
        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL3_ProcessEvent(&event);
            if (event.type == SDL_EVENT_QUIT) running = false;
            if (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_ESCAPE) running = false;
            if (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_R &&
                (event.key.mod & SDL_KMOD_GUI)) {
                running = false;
                restart = true;
            }
            if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN && event.button.button == SDL_BUTTON_LEFT &&
                event.button.clicks == 2 && !ImGui::GetIO().WantCaptureMouse) {
                pick_requested = true;
                pick_x = event.button.x;
                pick_y = event.button.y;
            }
            events.push_back(event);
        }

        int win_w, win_h;
        SDL_GetWindowSize(window, &win_w, &win_h);

        // --- Timing ---
        float time = (float)(SDL_GetPerformanceCounter() - start_time) / (float)freq;
        float dt = time - prev_time;
        prev_time = time;

        // --- Hot reload ---
        bool reloaded = shaders.poll_and_reload();

        // --- Depth readback → surface distance for speed scaling ---
        // Min of last frame's center patch; misses write max_dist (100) so an
        // all-sky patch just means "far away". Smoothed to absorb frame noise.
        {
            const float* d = (const float*)backend.buffer_contents(buf_depth_read);
            float min_d = 1e9f;
            for (uint32_t i = 0; i < DEPTH_PATCH * DEPTH_PATCH; i++) {
                if (d[i] > 0 && d[i] < min_d) min_d = d[i];
            }
            if (min_d < 1e9f) {
                if (camera.nav_distance <= 0) camera.nav_distance = min_d;
                else camera.nav_distance += (min_d - camera.nav_distance) * (1.0f - std::exp(-dt * 10.0f));
            }
        }

        // --- Double-click: re-center trackball on the clicked surface point ---
        // Depth is ray-distance t, so the picked point is pos + ray_dir * t with
        // the ray built exactly like make_camera_ray in common.metal.
        if (pick_requested && camera.mode == CameraMode::Trackball) {
            const float* d = (const float*)backend.buffer_contents(buf_pick_read);
            float t = 1e9f;
            for (uint32_t i = 0; i < PICK_PATCH * PICK_PATCH; i++) {
                if (d[i] > 0 && d[i] < t) t = d[i];
            }
            if (t < 99.0f) {  // skip sky (miss writes max_dist = 100)
                float px = pick_x * (float)w / (float)win_w;
                float py = pick_y * (float)h / (float)win_h;
                float ndc_x = (px / (float)w) * 2.0f - 1.0f;
                float ndc_y = -((py / (float)h) * 2.0f - 1.0f);
                float half_fov = std::tan(camera.fov * 0.5f);
                float aspect = (float)w / (float)h;

                float fwd[3], up[3], right[3], dir[3];
                camera.get_vectors(fwd, up, right);
                v3::mad(fwd, right, ndc_x * half_fov * aspect, dir);
                v3::mad(dir, up, ndc_y * half_fov, dir);
                v3::normalize(dir, dir);
                v3::mad(camera.pos, dir, t, camera.target);
            }
        }

        // --- Camera update ---
        camera.update(dt, events.data(), (int)events.size(), (float)w, (float)h);

        // --- Handle resize ---
        uint32_t new_w = backend.drawable_width();
        uint32_t new_h = backend.drawable_height();
        bool resized = (new_w != w || new_h != h);
        if (resized) {
            w = new_w; h = new_h;
            half_w = w / 2; half_h = h / 2;
            backend.resize_texture(tex_current_color, half_w, half_h);
            backend.resize_texture(tex_current_depth, half_w, half_h);
            backend.resize_texture(tex_output, w, h);
            backend.resize_texture(tex_history_a, w, h);
            backend.resize_texture(tex_history_b, w, h);
            backend.resize_texture(tex_recon_depth_a, w, h);
            backend.resize_texture(tex_recon_depth_b, w, h);
        }

        // --- Build uniforms ---
        FrameUniforms uniforms = {};
        uniforms.time = time;
        uniforms.delta_time = dt;
        uniforms.frame_index = frame_index;
        uniforms.flags = camera.show_grid ? 1u : 0u;
        uniforms.resolution[0] = (float)w;
        uniforms.resolution[1] = (float)h;
        uniforms.inv_resolution[0] = 1.0f / (float)w;
        uniforms.inv_resolution[1] = 1.0f / (float)h;

        // Mouse
        float mx, my;
        SDL_GetMouseState(&mx, &my);
        uniforms.mouse[0] = mx / (float)w;
        uniforms.mouse[1] = my / (float)h;

        // Camera vectors
        std::memcpy(uniforms.camera_pos, camera.pos, sizeof(float) * 3);
        float fwd[3], up[3], right[3];
        camera.get_vectors(fwd, up, right);
        std::memcpy(uniforms.camera_fwd, fwd, sizeof(float) * 3);
        std::memcpy(uniforms.camera_up, up, sizeof(float) * 3);
        std::memcpy(uniforms.camera_right, right, sizeof(float) * 3);
        uniforms.camera_fov = camera.fov;

        // View-projection matrices
        float vp[16];
        camera.get_view_proj((float)w / (float)h, 0.001f, 1000.0f, vp);
        std::memcpy(uniforms.view_proj, vp, sizeof(float) * 16);
        std::memcpy(uniforms.prev_view_proj, prev_vp, sizeof(float) * 16);

        float inv_vp[16];
        if (mat4::invert(vp, inv_vp)) {
            std::memcpy(uniforms.inv_view_proj, inv_vp, sizeof(float) * 16);
        }
        std::memcpy(prev_vp, vp, sizeof(float) * 16);

        // Jitter (Halton 2,3)
        uniforms.jitter[0] = halton(frame_index, 2) - 0.5f;
        uniforms.jitter[1] = halton(frame_index, 3) - 0.5f;

        // Pack raymarch params
        const auto& rm_params = shaders.get_params("raymarch.metal");
        uniforms.param_count = (uint32_t)rm_params.size();
        for (int i = 0; i < (int)rm_params.size() && i < 32; i++) {
            std::memcpy(uniforms.params[i], rm_params[i].current_val, sizeof(float) * 4);
        }

        // Per-iteration IFS rotation, hoisted out of the DE (params[3] = rotation)
        {
            static const float X[3] = {1, 0, 0}, Y[3] = {0, 1, 0}, Z[3] = {0, 0, 1};
            float rx[9], ry[9], rz[9], rxy[9], rot[9];
            mat3::axis_rotation(X, uniforms.params[3][0], rx);
            mat3::axis_rotation(Y, uniforms.params[3][1], ry);
            mat3::axis_rotation(Z, uniforms.params[3][2], rz);
            mat3::multiply(rx, ry, rxy);
            mat3::multiply(rxy, rz, rot);
            for (int col = 0; col < 3; col++) {
                std::memcpy(uniforms.rot_mtx[col], rot + col * 3, sizeof(float) * 3);
                uniforms.rot_mtx[col][3] = 0;
            }
        }

        // Pack reconstruct params
        const auto& rc_params = shaders.get_params("reconstruct.metal");
        uniforms.recon_param_count = (uint32_t)rc_params.size();
        for (int i = 0; i < (int)rc_params.size() && i < 8; i++) {
            std::memcpy(uniforms.recon_params[i], rc_params[i].current_val, sizeof(float) * 4);
        }

        // Pack pathtrace params
        const auto& pt_params = shaders.get_params("pathtrace.metal");
        uniforms.pt_param_count = (uint32_t)pt_params.size();
        for (int i = 0; i < (int)pt_params.size() && i < 16; i++) {
            std::memcpy(uniforms.pt_params[i], pt_params[i].current_val, sizeof(float) * 4);
        }

        // --- Accumulation counter ---
        // Hard changes (params, reload, resize, mode) restart from scratch.
        // Camera motion only caps the history weight at ~1/taa_alpha frames:
        // reprojection keeps history valid, so accumulation resumes from the
        // moving preview instead of resetting when the camera stops.
        bool camera_moved = std::memcmp(uniforms.camera_pos, prev_uniforms.camera_pos,
            offsetof(FrameUniforms, jitter) - offsetof(FrameUniforms, camera_pos)) != 0;
        bool params_changed = std::memcmp(uniforms.params, prev_uniforms.params,
            sizeof(uniforms.params) + sizeof(uniforms.recon_params) + sizeof(uniforms.pt_params)) != 0;
        bool hard_reset = params_changed || reloaded || resized || renderer_mode != prev_renderer_mode;

        // Converged-frame throttle: a sample's marginal value falls as 1/N,
        // so once accumulation is deep, back off — sample every (N/256)-th
        // frame, capped at 1/64. N then grows ~sqrt(wall time) instead of
        // pinning the GPU forever. GUI, readbacks, and blit still run every
        // frame; any invalidation resumes full rate instantly.
        constexpr uint32_t THROTTLE_START = 256;
        constexpr uint32_t THROTTLE_MAX_INTERVAL = 64;
        uint32_t sample_interval = std::clamp(accum_frames / THROTTLE_START, 1u, THROTTLE_MAX_INTERVAL);
        bool skip_sampling = !hard_reset && !camera_moved && sample_interval > 1 &&
                             (frame_index % sample_interval) != 0;

        if (hard_reset) {
            accum_frames = 0;
        } else if (camera_moved) {
            float taa_alpha = uniforms.recon_params[3][0];  // slot 3: taa_alpha
            uint32_t preview_cap = (uint32_t)std::max(1.0f / std::max(taa_alpha, 0.02f) - 1.0f, 0.0f);
            accum_frames = std::min(accum_frames + 1, preview_cap);
        } else if (!skip_sampling) {
            accum_frames++;
        }
        uniforms.accum_frames = accum_frames;

        // Sparse PT sampling: stride k traces 1 of k*k half-res pixels per
        // frame, round-robin. Keep the clamped TAA path (moving flag) until
        // every texel has been re-traced after motion stops — stale texels
        // from mid-motion frames must not enter the unclamped accumulator.
        //
        // While the camera moves (and until the post-motion refresh
        // completes), sparsity is capped at 2x2: stale texels smear under
        // motion, and reconstruct's 3x3 window is only guaranteed a fresh
        // tap for strides <= 3. Reconstruct also down-weights stale taps by
        // age (taa_stale_penalty).
        uint32_t slider_stride = 1;
        if (renderer_mode == 1) {
            slider_stride = (uint32_t)std::clamp((int)uniforms.pt_params[4][0], 1, 4);  // slot 4: sample_stride
        }
        uint32_t moving_stride = std::min(slider_stride, 2u);
        frames_since_move = camera_moved ? 0 : frames_since_move + 1;
        bool moving_window = frames_since_move <= moving_stride * moving_stride;
        uint32_t pt_stride = moving_window ? moving_stride : slider_stride;
        if (moving_window) uniforms.flags |= 2u;

        prev_uniforms = uniforms;
        prev_renderer_mode = renderer_mode;

        // Hard resets wipe history, so every texel must be freshly traced
        // that frame — stale ones would ghost the old scene in. The
        // effective stride is packed after the prev_uniforms snapshot (which
        // keeps the slider value) so overrides can't retrigger a reset.
        if (accum_frames == 0) pt_stride = 1;
        uniforms.pt_params[4][0] = (float)pt_stride;

        int buf_u = buf_uniforms[frame_index % 3];
        backend.write_buffer(buf_u, &uniforms, sizeof(uniforms));

        // --- Render ---
        backend.begin_frame();

        // Pass 1: Raymarch or path trace (half-res)
        int rm_pipeline = renderer_mode == 1 ? shaders.get_pipeline("pathtrace_kernel")
                                             : shaders.get_pipeline("raymarch_kernel");
        if (rm_pipeline >= 0 && !skip_sampling) {
            backend.dispatch({
                .pipeline_id = rm_pipeline,
                .grid_width = (half_w + pt_stride - 1) / pt_stride,
                .grid_height = (half_h + pt_stride - 1) / pt_stride,
                .threadgroup_w = 16,
                .threadgroup_h = 16,
                .textures = {tex_current_color, tex_current_depth},
                .buffers = {buf_u}
            });
        }

        // Copy center depth patch for CPU readback next frame
        backend.copy_texture_to_buffer(tex_current_depth, buf_depth_read,
            half_w / 2 - DEPTH_PATCH / 2, half_h / 2 - DEPTH_PATCH / 2,
            DEPTH_PATCH, DEPTH_PATCH, sizeof(float));

        // Copy depth patch under the cursor for pick readback next frame
        uint32_t pick_tx = (uint32_t)std::clamp((int)(mx * (float)w / (float)win_w * 0.5f) - (int)(PICK_PATCH / 2),
                                                0, (int)(half_w - PICK_PATCH));
        uint32_t pick_ty = (uint32_t)std::clamp((int)(my * (float)h / (float)win_h * 0.5f) - (int)(PICK_PATCH / 2),
                                                0, (int)(half_h - PICK_PATCH));
        backend.copy_texture_to_buffer(tex_current_depth, buf_pick_read,
            pick_tx, pick_ty, PICK_PATCH, PICK_PATCH, sizeof(float));

        // Pass 2: Reconstruct (full-res, half → full with joint-bilateral)
        int rc_pipeline = shaders.get_pipeline("reconstruct_kernel");
        int history_read  = ping ? tex_history_a : tex_history_b;
        int history_write = ping ? tex_history_b : tex_history_a;
        int depth_read    = ping ? tex_recon_depth_a : tex_recon_depth_b;
        int depth_write   = ping ? tex_recon_depth_b : tex_recon_depth_a;
        if (rc_pipeline >= 0 && !skip_sampling) {
            backend.dispatch({
                .pipeline_id = rc_pipeline,
                .grid_width = w,
                .grid_height = h,
                .threadgroup_w = 16,
                .threadgroup_h = 16,
                .textures = {tex_current_color, tex_current_depth, history_read, history_write, depth_read, depth_write},
                .buffers = {buf_u}
            });
        }

        // Pass 3: Present (full-res passthrough stub). Skipped on throttled
        // frames — tex_output keeps the last rendered image for the blit.
        int present_pipeline = shaders.get_pipeline("present_kernel");
        if (present_pipeline >= 0 && !skip_sampling) {
            backend.dispatch({
                .pipeline_id = present_pipeline,
                .grid_width = w,
                .grid_height = h,
                .threadgroup_w = 16,
                .threadgroup_h = 16,
                .textures = {history_write, tex_output},
                .buffers = {buf_u}
            });
        }

        // Only advance the ping-pong if reconstruct actually ran — otherwise
        // (skipped frame or mid-edit compile error) present would read an
        // unwritten texture next time.
        if (rc_pipeline >= 0 && !skip_sampling) ping = !ping;

        // Blit to screen
        backend.blit_to_screen(tex_output);

        // ImGui
        backend.imgui_new_frame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();

        // Trackball pivot marker: project target to screen, draw behind ImGui windows
        if (camera.mode == CameraMode::Trackball) {
            float clip[4];
            for (int r = 0; r < 4; r++) {
                clip[r] = vp[r] * camera.target[0] + vp[4 + r] * camera.target[1] +
                          vp[8 + r] * camera.target[2] + vp[12 + r];
            }
            if (clip[3] > 0) {
                ImVec2 disp = ImGui::GetIO().DisplaySize;
                ImVec2 c((clip[0] / clip[3] * 0.5f + 0.5f) * disp.x,
                         (0.5f - clip[1] / clip[3] * 0.5f) * disp.y);
                ImDrawList* dl = ImGui::GetBackgroundDrawList();
                dl->AddCircle(c, 9.0f, IM_COL32(0, 0, 0, 200), 0, 3.5f);
                dl->AddCircle(c, 9.0f, IM_COL32(255, 170, 40, 255), 0, 1.5f);
                dl->AddCircleFilled(c, 2.0f, IM_COL32(255, 170, 40, 255));
            }
        }

        ImGui::Begin("Fractal Engine");
        {
            static constexpr int FT_SAMPLES = 120;
            static float ft_buf[FT_SAMPLES] = {0};
            static int ft_idx = 0;
            ft_buf[ft_idx] = dt * 1000.0f;
            ft_idx = (ft_idx + 1) % FT_SAMPLES;
            char overlay[64];
            snprintf(overlay, sizeof(overlay), "%.1f fps (%.2f ms)", 1.0f / dt, dt * 1000.0f);
            ImGui::PlotLines("##frametime", ft_buf, FT_SAMPLES, ft_idx, overlay,
                             0.0f, 33.3f, ImVec2(0, 40));

            static float gpu_buf[FT_SAMPLES] = {0};
            static int gpu_idx = 0;
            float gpu_ms = backend.gpu_frame_ms();
            gpu_buf[gpu_idx] = gpu_ms;
            gpu_idx = (gpu_idx + 1) % FT_SAMPLES;
            snprintf(overlay, sizeof(overlay), "GPU %.2f ms", gpu_ms);
            ImGui::PlotLines("##gputime", gpu_buf, FT_SAMPLES, gpu_idx, overlay,
                             0.0f, 33.3f, ImVec2(0, 40));
        }
        ImGui::Text("Resolution: %u x %u", w, h);
        {
            const char* renderer_names[] = {"Raymarch", "Path Trace"};
            ImGui::Combo("Renderer", &renderer_mode, renderer_names, IM_ARRAYSIZE(renderer_names));
        }
        if (sample_interval > 1) {
            ImGui::Text("Accumulated: %u frames (throttled to 1/%u)", accum_frames, sample_interval);
        } else {
            ImGui::Text("Accumulated: %u frames", accum_frames);
        }
        ImGui::Checkbox("Show Grid", &camera.show_grid);
        render_shader_errors(shaders);

        if (ImGui::CollapsingHeader("Camera")) {
            const char* mode_names[] = {"Trackball", "FPS"};
            int mode_idx = (int)camera.mode;
            if (ImGui::Combo("Mode", &mode_idx, mode_names, IM_ARRAYSIZE(mode_names))) {
                camera.mode = (CameraMode)mode_idx;
            }
            ImGui::Text("Pos:    %.2f, %.2f, %.2f", camera.pos[0], camera.pos[1], camera.pos[2]);
            ImGui::Text("Target: %.2f, %.2f, %.2f", camera.target[0], camera.target[1], camera.target[2]);
            ImGui::Text("Fwd:    %.3f, %.3f, %.3f", fwd[0], fwd[1], fwd[2]);
            ImGui::Text("Up:     %.3f, %.3f, %.3f", up[0], up[1], up[2]);
            ImGui::Text("Right:  %.3f, %.3f, %.3f", right[0], right[1], right[2]);
            {
                float off[3]; v3::sub(camera.pos, camera.target, off);
                ImGui::Text("Distance: %.2f", v3::length(off));
            }
            ImGui::Text("Surface Dist: %.4f", camera.nav_distance);
            ImGui::Checkbox("Adaptive Speed", &camera.adaptive_speed);
            ImGui::SliderFloat("Rotate Speed", &camera.rotate_speed, 0.01f, 5.0f);
            ImGui::SliderFloat("Pan Speed", &camera.pan_speed, 0.01f, 5.0f);
            ImGui::SliderFloat("Zoom Speed", &camera.zoom_speed, 0.01f, 1.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("Keyboard Speed", &camera.keyboard_speed, 0.01f, 2.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
        }

        if (ImGui::CollapsingHeader("Shader Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
            render_shader_params(shaders.get_params_mut("raymarch.metal"));
        }
        if (renderer_mode == 1 && ImGui::CollapsingHeader("Path Tracer", ImGuiTreeNodeFlags_DefaultOpen)) {
            render_shader_params(shaders.get_params_mut("pathtrace.metal"));
        }
        if (ImGui::CollapsingHeader("Reconstruction", ImGuiTreeNodeFlags_DefaultOpen)) {
            render_shader_params(shaders.get_params_mut("reconstruct.metal"));
        }
        ImGui::End();

        ImGui::Render();
        backend.render_imgui();

        backend.end_frame();

        // Auto-save state if anything changed
        state.save_if_changed(shaders, time);

        frame_index++;
    }

    // Cleanup — force a save so debounced changes survive quit/restart
    state.save(shaders);
    backend.imgui_shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
    backend.shutdown();
    SDL_DestroyWindow(window);
    SDL_Quit();

    if (restart) {
        char* const args[] = {(char*)exe_path.c_str(), nullptr};
        execv(exe_path.c_str(), args);
        std::cerr << "[restart] execv failed: " << exe_path << "\n";
        return 1;
    }
    return 0;
}
