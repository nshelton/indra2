#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "metal_backend.h"
#include "shader_manager.h"
#include "gui.h"
#include "math_util.h"
#include "state_serializer.h"
#include <cstring>
#include <vector>
#include <iostream>

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
    shaders.register_shader("reconstruct.metal", "reconstruct_kernel");
    shaders.register_shader("present.metal", "present_kernel");

    // 5. Create textures
    uint32_t w = backend.drawable_width();
    uint32_t h = backend.drawable_height();
    uint32_t half_w = w / 2, half_h = h / 2;

    int tex_current_color = backend.create_texture({half_w, half_h, TextureDesc::RGBA16Float, true, "current_color"});
    int tex_current_depth = backend.create_texture({half_w, half_h, TextureDesc::R32Float, true, "current_depth"});
    int tex_output        = backend.create_texture({w, h, TextureDesc::RGBA16Float, true, "output"});
    int tex_history_a     = backend.create_texture({w, h, TextureDesc::RGBA16Float, true, "history_a"});
    int tex_history_b     = backend.create_texture({w, h, TextureDesc::RGBA16Float, true, "history_b"});
    int tex_reconstructed_depth = backend.create_texture({w, h, TextureDesc::R32Float, true, "reconstructed_depth"});
    bool ping = false;

    // 6. Create uniform buffer
    int buf_uniforms = backend.create_buffer(sizeof(FrameUniforms), "uniforms");

    // 7. Camera
    Camera camera;

    // 8. State persistence — load saved state
    std::string state_path = std::string(SDL_GetBasePath()) + "state.json";
    StateSerializer state(state_path);
    state.load(camera, shaders);

    // 9. Timing
    uint64_t start_time = SDL_GetPerformanceCounter();
    uint64_t freq = SDL_GetPerformanceFrequency();
    uint32_t frame_index = 0;
    float prev_time = 0;
    float prev_vp[16];
    mat4::identity(prev_vp);

    // Main loop
    bool running = true;
    while (running) {
        // --- Events ---
        SDL_Event event;
        std::vector<SDL_Event> events;
        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL3_ProcessEvent(&event);
            if (event.type == SDL_EVENT_QUIT) running = false;
            if (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_ESCAPE) running = false;
            events.push_back(event);
        }

        // --- Timing ---
        float time = (float)(SDL_GetPerformanceCounter() - start_time) / (float)freq;
        float dt = time - prev_time;
        prev_time = time;

        // --- Hot reload ---
        shaders.poll_and_reload();

        // --- Camera update ---
        camera.update(dt, events.data(), (int)events.size(), (float)w, (float)h);

        // --- Handle resize ---
        uint32_t new_w = backend.drawable_width();
        uint32_t new_h = backend.drawable_height();
        if (new_w != w || new_h != h) {
            w = new_w; h = new_h;
            half_w = w / 2; half_h = h / 2;
            backend.resize_texture(tex_current_color, half_w, half_h);
            backend.resize_texture(tex_current_depth, half_w, half_h);
            backend.resize_texture(tex_output, w, h);
            backend.resize_texture(tex_history_a, w, h);
            backend.resize_texture(tex_history_b, w, h);
            backend.resize_texture(tex_reconstructed_depth, w, h);
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

        // Pack reconstruct params
        const auto& rc_params = shaders.get_params("reconstruct.metal");
        uniforms.recon_param_count = (uint32_t)rc_params.size();
        for (int i = 0; i < (int)rc_params.size() && i < 8; i++) {
            std::memcpy(uniforms.recon_params[i], rc_params[i].current_val, sizeof(float) * 4);
        }

        backend.write_buffer(buf_uniforms, &uniforms, sizeof(uniforms));

        // --- Render ---
        backend.begin_frame();

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
                .textures = {tex_current_color, tex_current_depth, history_read, history_write, tex_reconstructed_depth},
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

        // ImGui
        backend.imgui_new_frame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Fractal Engine");
        ImGui::Text("%.1f fps (%.2f ms)", 1.0f / dt, dt * 1000.0f);
        ImGui::Text("Resolution: %u x %u", w, h);
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
            ImGui::SliderFloat("Rotate Speed", &camera.rotate_speed, 0.01f, 5.0f);
            ImGui::SliderFloat("Pan Speed", &camera.pan_speed, 0.01f, 5.0f);
            ImGui::SliderFloat("Zoom Speed", &camera.zoom_speed, 0.01f, 2.0f);
            ImGui::SliderFloat("Keyboard Speed", &camera.keyboard_speed, 0.1f, 5.0f);
        }

        if (ImGui::CollapsingHeader("Shader Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
            render_shader_params(shaders.get_params_mut("raymarch.metal"));
        }
        if (ImGui::CollapsingHeader("Reconstruction", ImGuiTreeNodeFlags_DefaultOpen)) {
            render_shader_params(shaders.get_params_mut("reconstruct.metal"));
        }
        ImGui::End();

        ImGui::Render();
        backend.render_imgui();

        backend.end_frame();

        // Auto-save state if anything changed
        state.save_if_changed(camera, shaders, time);

        frame_index++;
    }

    // Cleanup
    backend.imgui_shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
    backend.shutdown();
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
