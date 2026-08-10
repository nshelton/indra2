#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "metal_backend.h"
#include "shader_manager.h"
#include "gui.h"
#include "math_util.h"
#include "state_serializer.h"
#include "timeline.h"
#include "render_job.h"
#include "capture.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <vector>
#include <iostream>
#include <unistd.h>

// A destructive preset action, deferred from the click to a confirmation.
// Deferred because the row buttons live inside ImGui::PushID(name): calling
// OpenPopup there would not match a BeginPopupModal drawn at window scope, and
// the dialog would silently never appear.
struct PendingPreset {
    enum Kind { None, Load, Update, Delete };
    Kind kind = None;
    std::string name;
};

static const char* preset_popup_title(PendingPreset::Kind k) {
    switch (k) {
        case PendingPreset::Load:   return "Unsaved animation";
        case PendingPreset::Update: return "Overwrite preset?";
        case PendingPreset::Delete: return "Delete preset?";
        default:                    return "";
    }
}

// SDL's folder dialog delivers its result on an arbitrary thread; hand it to
// the UI thread through a release/acquire flag. filelist[0] is the picked
// path; null filelist (error) or empty list (cancel) leave everything as-is.
static char s_picked_dir[256];
static std::atomic<bool> s_picked_ready{false};
static void on_render_dir_picked(void*, const char* const* filelist, int) {
    if (filelist && filelist[0]) {
        SDL_strlcpy(s_picked_dir, filelist[0], sizeof(s_picked_dir));
        s_picked_ready.store(true, std::memory_order_release);
    }
}

// "51s", "4m51s", "10m", "1h4m" — the two largest nonzero units. An ETA is
// a glance value; nobody wants to divide 600 by 60 mid-render.
static void format_duration(double secs, char* out, size_t n) {
    int t = (int)std::lround(std::max(secs, 0.0));
    int h = t / 3600, m = (t / 60) % 60, s = t % 60;
    if (h > 0) {
        if (m > 0) std::snprintf(out, n, "%dh%dm", h, m);
        else       std::snprintf(out, n, "%dh", h);
    } else if (m > 0) {
        if (s > 0) std::snprintf(out, n, "%dm%ds", m, s);
        else       std::snprintf(out, n, "%dm", m);
    } else {
        std::snprintf(out, n, "%ds", s);
    }
}

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
    apply_gui_style();
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

    // 6. Uniform buffers — one slot per in-flight frame, since a single shared
    // buffer would let the CPU overwrite uniforms mid-read (torn matrices,
    // off-by-a-frame reprojection). begin_frame blocks once the pipeline is
    // full, so a slot is always retired before it is reused.
    constexpr int RING = MetalBackend::MAX_FRAMES_IN_FLIGHT;
    int buf_uniforms[RING];
    for (int i = 0; i < RING; i++) {
        buf_uniforms[i] = backend.create_buffer(sizeof(FrameUniforms), "uniforms" + std::to_string(i));
    }

    // Depth readback: center patch of the half-res depth texture, read one frame later
    constexpr uint32_t DEPTH_PATCH = 16;
    int buf_depth_read = backend.create_buffer(DEPTH_PATCH * DEPTH_PATCH * sizeof(float), "depth_readback");

    // Pick readback: depth patch under the cursor, for double-click trackball centering
    constexpr uint32_t PICK_PATCH = 4;
    int buf_pick_read = backend.create_buffer(PICK_PATCH * PICK_PATCH * sizeof(float), "pick_readback");

    // 7. Camera + renderer mode + animation
    Camera camera;
    int renderer_mode = 1;  // 0 = raymarch, 1 = path trace

    // The timeline owns any param it has a track for, and the camera when it
    // has camera keys — apply() writes straight into ShaderParam::current_val
    // and Camera, so every downstream system (the sliders, the params_changed
    // memcmp, serialization) is unchanged. Declared here so `enabled` can be
    // registered with the other persisted state below.
    Timeline timeline;
    TimelineUI timeline_ui;
    RenderJob job;

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
    // Session-only: panel visibility is an app preference. Inheriting it from
    // a preset would silently switch the timeline on and hand it the camera.
    state.session_bool_field("/timeline/enabled", &timeline.enabled);
    // Panel layout is a workspace preference, not part of a look — session only.
    state.session_bool_field("/gui/compact_params", &g_compact_param_labels);
    state.load(shaders);

    // Preset store: one full-state JSON per file, name = filename
    std::string preset_dir = std::string(SDL_GetBasePath()) + "presets/";
    std::vector<std::string> presets;
    auto refresh_presets = [&]() {
        presets.clear();
        std::error_code ec;
        for (const auto& e : std::filesystem::directory_iterator(preset_dir, ec)) {
            if (e.path().extension() == ".json") presets.push_back(e.path().stem().string());
        }
        std::sort(presets.begin(), presets.end());
    };
    std::filesystem::create_directories(preset_dir);
    refresh_presets();

    // Which preset the current state came from, and the timeline revision at
    // that moment. Dirty = the animation has been edited since. Parameter and
    // camera edits deliberately do not count: the camera moves constantly
    // while noodling, so including it would prompt on every single Load.
    //
    // Not persisted — StateSerializer has no string field, and after a restart
    // animations/_working.json restores the animation itself anyway.
    std::string current_preset;
    uint64_t preset_anim_revision = 0;

    auto load_preset = [&](const std::string& name) {
        if (!state.load_from(preset_dir + name + ".json", shaders, &timeline)) {
            std::cerr << "[preset] failed to load " << name << "\n";
            return;
        }
        current_preset = name;
        preset_anim_revision = timeline.revision;
        // The old playhead means nothing against a new animation.
        timeline.playhead = 0.0f;
        timeline.playing = false;
    };
    auto save_preset = [&](const std::string& name) {
        state.save_to(preset_dir + name + ".json", shaders, &timeline);
        current_preset = name;
        preset_anim_revision = timeline.revision;
    };
    auto anim_dirty = [&]() {
        return timeline.revision != preset_anim_revision &&
               (!timeline.tracks.empty() || !timeline.cam_keys.empty());
    };
    camera.zoom_speed = std::clamp(camera.zoom_speed, 0.01f, 1.0f);
    camera.keyboard_speed = std::clamp(camera.keyboard_speed, 0.01f, 2.0f);

    // 8b. Animation storage.
    std::string anim_dir = std::string(SDL_GetBasePath()) + "animations/";
    std::filesystem::create_directories(anim_dir);
    std::vector<std::string> anims;
    auto refresh_anims = [&]() {
        anims.clear();
        std::error_code ec;
        for (const auto& e : std::filesystem::directory_iterator(anim_dir, ec)) {
            if (e.path().extension() == ".json") anims.push_back(e.path().stem().string());
        }
        std::sort(anims.begin(), anims.end());
    };
    refresh_anims();

    // Debounced autosave of the working animation, mirroring StateSerializer's
    // dirty/debounce pattern but keyed on a revision counter — a memcmp over a
    // variable-length key list isn't meaningful.
    std::string anim_autosave = anim_dir + "_working.json";
    timeline.load(anim_autosave);
    uint64_t anim_saved_revision = timeline.revision;
    float anim_dirty_since = -1.0f;

    // Full-frame capture staging. Metal needs texture->buffer blit rows
    // 256-byte aligned; only the small patch readbacks satisfy that by luck.
    int buf_capture = -1;
    uint32_t capture_w = 0, capture_h = 0;
    size_t capture_row = 0;

    // Render resolution while a job is active (render scale), else the drawable.
    uint32_t job_w = 0, job_h = 0;

    // Target of an open preset confirmation dialog; spans frames.
    PendingPreset pending;

    // 9. Timing. Wall clock drives dt and the UI; animation time is separate
    // and, during an offline render, pinned per output frame.
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
    uint32_t trace_index = 0;  // advances only on frames that actually trace
    FrameUniforms prev_uniforms = {};

    std::string exe_path = std::string(SDL_GetBasePath()) + "fractal-engine";

    // Ends an offline render, whether it finished or was cancelled. Restores
    // vsync; the render-scale resize unwinds via the resize check below once
    // job.active is false.
    auto end_job = [&]() {
        job.active = false;
        job.capture_pending = false;
        backend.set_vsync(true);
    };

    // Main loop
    bool running = true;
    bool restart = false;
    while (running) {
        // --- Events ---
        SDL_Event event;
        std::vector<SDL_Event> events;
        bool pick_requested = false;
        bool esc_pressed = false;
        bool space_pressed = false;
        float pick_x = 0, pick_y = 0;
        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL3_ProcessEvent(&event);
            if (event.type == SDL_EVENT_QUIT) running = false;
            // Not while a text field has focus — Esc is "cancel this edit"
            // there, and the param range popup puts input boxes a right-click
            // away from anywhere in the panel.
            if (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_ESCAPE &&
                !ImGui::GetIO().WantCaptureKeyboard) {
                esc_pressed = true;
            }
            if (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_SPACE && !event.key.repeat) {
                space_pressed = true;
            }
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

        // Esc cancels an offline render before it quits the app — a long
        // render is exactly when you'd hit it by reflex.
        if (esc_pressed) {
            if (job.active) end_job();
            else running = false;
        }

        // Space toggles playback. Gated on WantTextInput rather than
        // WantCaptureKeyboard so it still works with a panel focused, but
        // typing a preset or output-directory name inserts a space as normal.
        if (space_pressed && timeline.enabled && !job.active &&
            !ImGui::GetIO().WantTextInput) {
            timeline.playing = !timeline.playing;
        }

        // macOS 26 sometimes drops the button-up mid-drag (SDL #12218/#15967),
        // leaving ImGui's slider stuck to the captured mouse. If the OS says
        // the button is up but ImGui still thinks it's down, inject the release.
        {
            SDL_MouseButtonFlags held = SDL_GetGlobalMouseState(nullptr, nullptr);
            ImGuiIO& io = ImGui::GetIO();
            if (!(held & SDL_BUTTON_LMASK) && io.MouseDown[0]) io.AddMouseButtonEvent(0, false);
            if (!(held & SDL_BUTTON_RMASK) && io.MouseDown[1]) io.AddMouseButtonEvent(1, false);
        }

        int win_w, win_h;
        SDL_GetWindowSize(window, &win_w, &win_h);

        // --- Timing ---
        float time = (float)(SDL_GetPerformanceCounter() - start_time) / (float)freq;
        float dt = time - prev_time;
        prev_time = time;

        // --- Hot reload ---
        bool reloaded = shaders.poll_and_reload();

        // --- Animation time ---
        // Offline: pinned to the output frame, offset within the shutter by
        // the current accumulation sample. Interactive: wall clock while
        // playing, otherwise wherever the playhead was left.
        float anim_time = timeline.playhead;
        if (job.active) {
            anim_time = ((float)job.frame + job.shutter_offset()) / timeline.fps;
            timeline.playhead = (float)job.frame / timeline.fps;
        } else if (timeline.playing) {
            timeline.playhead += dt;
            if (timeline.playhead >= timeline.duration) {
                timeline.playhead = timeline.loop ? 0.0f : timeline.duration;
                if (!timeline.loop) timeline.playing = false;
            }
            anim_time = timeline.playhead;
        }
        if (!timeline.enabled) timeline.playing = false;
        timeline.apply(anim_time, shaders, camera);

        // The timeline owns the camera unless auto-key is on (where manual
        // moves become keys) or a render is in flight (where nothing may touch
        // the pose mid-frame).
        bool cam_locked = job.active || (timeline.camera_driven() && !timeline.auto_key);

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
        if (pick_requested && camera.mode == CameraMode::Trackball && !cam_locked) {
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
        if (!cam_locked) {
            float pre_pos[3], pre_target[3], pre_fov = camera.fov;
            std::memcpy(pre_pos, camera.pos, sizeof(pre_pos));
            std::memcpy(pre_target, camera.target, sizeof(pre_target));

            camera.update(dt, events.data(), (int)events.size(), (float)w, (float)h);

            // enabled check: a parked (disabled) timeline must not silently
            // collect keys from camera moves just because auto-key was left on.
            if (timeline.enabled && timeline.auto_key && timeline.drive_camera &&
                (std::memcmp(pre_pos, camera.pos, sizeof(pre_pos)) != 0 ||
                 std::memcmp(pre_target, camera.target, sizeof(pre_target)) != 0 ||
                 pre_fov != camera.fov)) {
                timeline.key_camera(timeline.playhead, camera);
            }
        }

        // --- Handle resize ---
        // A render job pins the resolution (render scale); clearing job.active
        // lets the next frame snap back to the drawable.
        uint32_t new_w = job.active ? job_w : backend.drawable_width();
        uint32_t new_h = job.active ? job_h : backend.drawable_height();
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
        // Authored time when the timeline is on, so a shader reading frame.time
        // animates with the playhead and renders reproducibly. Off, it falls
        // back to wall clock — a procedural shader should still move while you
        // are just noodling.
        uniforms.time = timeline.enabled ? anim_time : time;
        uniforms.delta_time = dt;
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

        // Pack raymarch params
        const auto& rm_params = shaders.get_params("raymarch.metal");
        uniforms.param_count = (uint32_t)rm_params.size();
        for (int i = 0; i < (int)rm_params.size() && i < 32; i++) {
            std::memcpy(uniforms.params[i], rm_params[i].current_val, sizeof(float) * 4);
        }

        // Per-iteration IFS rotation, hoisted out of the DE (params[2] = rotation)
        {
            static const float X[3] = {1, 0, 0}, Y[3] = {0, 1, 0}, Z[3] = {0, 0, 1};
            float rx[9], ry[9], rz[9], rxy[9], rot[9];
            mat3::axis_rotation(X, uniforms.params[2][0], rx);
            mat3::axis_rotation(Y, uniforms.params[2][1], ry);
            mat3::axis_rotation(Z, uniforms.params[2][2], rz);
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
        for (int i = 0; i < (int)rc_params.size() && i < 12; i++) {
            std::memcpy(uniforms.recon_params[i], rc_params[i].current_val, sizeof(float) * 4);
        }

        // Pack pathtrace params
        const auto& pt_params = shaders.get_params("pathtrace.metal");
        uniforms.pt_param_count = (uint32_t)pt_params.size();
        for (int i = 0; i < (int)pt_params.size() && i < 16; i++) {
            std::memcpy(uniforms.pt_params[i], pt_params[i].current_val, sizeof(float) * 4);
        }

        // Pack present params
        const auto& ps_params = shaders.get_params("present.metal");
        uniforms.post_param_count = (uint32_t)ps_params.size();
        for (int i = 0; i < (int)ps_params.size() && i < 8; i++) {
            std::memcpy(uniforms.post_params[i], ps_params[i].current_val, sizeof(float) * 4);
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

        // --- Offline render overrides ---
        // Within one output frame the params and camera DO change every sample
        // (shutter jitter), which would normally hard-reset and wipe the
        // accumulator. Suppressing that is what makes motion blur work: each
        // sample lands in reconstruct's unclamped running mean at a different
        // time inside the shutter, which integrates to ground-truth blur.
        //
        // camera_moved must be forced off too. It would otherwise cap
        // accumulation and switch reconstruct to the reprojecting,
        // neighborhood-clamped TAA path — that clamp biases every sample
        // toward the median pose, which is precisely the blur we're trying to
        // capture.
        if (job.active) {
            hard_reset = job.needs_reset || reloaded || resized;
            camera_moved = false;
            frames_since_move = 1000;   // keep the "moving" window closed
            job.needs_reset = false;
        }

        // Converged-frame throttle: a sample's marginal value falls as 1/N,
        // so once accumulation is deep, back off — sample every (N/256)-th
        // frame, capped at 1/64. N then grows ~sqrt(wall time) instead of
        // pinning the GPU forever. GUI, readbacks, and blit still run every
        // frame; any invalidation resumes full rate instantly.
        constexpr uint32_t THROTTLE_START = 256;
        constexpr uint32_t THROTTLE_MAX_INTERVAL = 64;
        uint32_t sample_interval = job.active ? 1u
            : std::clamp(accum_frames / THROTTLE_START, 1u, THROTTLE_MAX_INTERVAL);
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

        // Sample clock: advances only on frames that actually trace, so the
        // sparse round-robin keeps turning under the converged throttle. (A
        // real frame counter stalls on whichever cell aligns with the skip
        // interval, freezing the other texels' samples forever.)
        //
        // Offline, seed from (output frame, sample) instead: re-rendering a
        // frame then reproduces it exactly, while noise still decorrelates
        // between output frames. A single fixed pattern reused every frame
        // reads far worse in motion than noise that moves.
        uniforms.frame_index = job.active
            ? (uint32_t)(job.frame * (int)std::max(job.samples, 1u) + (int)job.sample)
            : trace_index;
        if (!skip_sampling) trace_index++;

        // R2 additive-recurrence jitter: any strided subsequence of an
        // irrational rotation stays equidistributed, so texels traced every
        // k*k-th sample still cover their full 2x2 quad. (Halton base 2
        // subsampled at a power-of-two stride collapses to a narrow slice —
        // visible aliasing at sample_stride 4.)
        uniforms.jitter[0] = (float)std::fmod(uniforms.frame_index * 0.7548776662466927, 1.0) - 0.5f;
        uniforms.jitter[1] = (float)std::fmod(uniforms.frame_index * 0.5698402909980532, 1.0) - 0.5f;

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

        // Offline: trace every texel every sample — sparsity trades noise for
        // interactivity, and there is no interactivity to buy here. An open
        // shutter also disables depth seeding (flag bit 2): the surface can
        // close in between samples by more than seed_primary_t's 10% back-off.
        if (job.active) {
            pt_stride = 1;
            if (job.shutter > 0.0f) uniforms.flags |= 4u;
        }

        prev_uniforms = uniforms;
        prev_renderer_mode = renderer_mode;

        // Hard resets wipe history, so every texel must be freshly traced
        // that frame — stale ones would ghost the old scene in. The
        // effective stride is packed after the prev_uniforms snapshot (which
        // keeps the slider value) so overrides can't retrigger a reset.
        if (accum_frames == 0) pt_stride = 1;
        uniforms.pt_params[4][0] = (float)pt_stride;

        // --- Render ---
        // begin_frame first: it blocks until the frame that last used this ring
        // slot has retired, so the write below cannot land in a buffer the GPU
        // is still reading.
        backend.begin_frame();

        int buf_u = buf_uniforms[frame_index % RING];
        backend.write_buffer(buf_u, &uniforms, sizeof(uniforms));

        // Ping-pong selection is shared by pass 1 and pass 2: depth_read is
        // last frame's reconstructed depth, which pathtrace uses to seed
        // primary rays (bread crumbs) and reconstruct uses for disocclusion.
        int history_read  = ping ? tex_history_a : tex_history_b;
        int history_write = ping ? tex_history_b : tex_history_a;
        int depth_read    = ping ? tex_recon_depth_a : tex_recon_depth_b;
        int depth_write   = ping ? tex_recon_depth_b : tex_recon_depth_a;

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
                .textures = {tex_current_color, tex_current_depth, depth_read},
                .buffers = {buf_u},
                .label = renderer_mode == 1 ? "pathtrace" : "raymarch"
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
        if (rc_pipeline >= 0 && !skip_sampling) {
            backend.dispatch({
                .pipeline_id = rc_pipeline,
                .grid_width = w,
                .grid_height = h,
                .threadgroup_w = 16,
                .threadgroup_h = 16,
                .textures = {tex_current_color, tex_current_depth, history_read, history_write, depth_read, depth_write},
                .buffers = {buf_u},
                .label = "reconstruct"
            });
        }

        // Only advance the ping-pong if reconstruct actually ran — otherwise
        // (skipped frame or mid-edit compile error) present would read an
        // unwritten texture next time.
        if (rc_pipeline >= 0 && !skip_sampling) ping = !ping;

        // Pass 3: Present (tonemap/grade). Runs every frame, even throttled
        // ones, so grading sliders respond instantly; after the flip,
        // ping ? a : b is the last history reconstruct actually wrote.
        int present_pipeline = shaders.get_pipeline("present_kernel");
        if (present_pipeline >= 0) {
            backend.dispatch({
                .pipeline_id = present_pipeline,
                .grid_width = w,
                .grid_height = h,
                .threadgroup_w = 16,
                .threadgroup_h = 16,
                .textures = {ping ? tex_history_a : tex_history_b, tex_output},
                .buffers = {buf_u},
                .label = "present"
            });
        }

        // Blit to screen
        backend.blit_to_screen(tex_output);

        // --- Offline capture ---
        // tex_output is the final display-ready image and ImGui draws into the
        // drawable, not into it, so the UI is excluded for free.
        if (job.active) {
            job.sample++;
            if (job.sample >= job.samples) {
                if (capture_w != w || capture_h != h) {
                    if (buf_capture >= 0) backend.destroy_buffer(buf_capture);
                    capture_row = aligned_row_bytes(w, 8);   // RGBA16Float
                    buf_capture = backend.create_buffer(capture_row * h, "capture");
                    capture_w = w;
                    capture_h = h;
                }
                if (buf_capture >= 0) {
                    backend.copy_texture_to_buffer(tex_output, buf_capture, 0, 0, w, h, 8,
                                                   (uint32_t)capture_row);
                    job.capture_pending = true;
                }
            }
        }

        // ImGui
        backend.imgui_new_frame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();

        // Re-latched by whichever param slider is held this frame; apply()
        // reads it next frame to leave that channel alone mid-drag.
        timeline.editing_shader.clear();
        timeline.editing_param.clear();

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

            // Smooth the readout text so it's legible; plots stay raw
            static float ft_avg = 0, gpu_avg = 0;
            float gpu_ms = backend.gpu_frame_ms();
            if (ft_avg == 0) ft_avg = dt * 1000.0f;
            ft_avg += (dt * 1000.0f - ft_avg) * 0.05f;
            gpu_avg += (gpu_ms - gpu_avg) * 0.05f;

            char overlay[64];
            snprintf(overlay, sizeof(overlay), "%.1f fps (%.2f ms)", 1000.0f / ft_avg, ft_avg);
            ImGui::PlotLines("##frametime", ft_buf, FT_SAMPLES, ft_idx, overlay,
                             0.0f, 33.3f, ImVec2(0, 40));

            static float gpu_buf[FT_SAMPLES] = {0};
            static int gpu_idx = 0;
            gpu_buf[gpu_idx] = gpu_ms;
            gpu_idx = (gpu_idx + 1) % FT_SAMPLES;
            snprintf(overlay, sizeof(overlay), "GPU %.2f ms", gpu_avg);
            ImGui::PlotLines("##gputime", gpu_buf, FT_SAMPLES, gpu_idx, overlay,
                             0.0f, 33.3f, ImVec2(0, 40));

            // Per-pass GPU breakdown (stage-boundary timestamps), smoothed.
            // Rows are drawn from the persistent label set, NOT the current
            // frame's list: the converged throttle skips trace/reconstruct
            // on most frames, and rows blinking in and out would re-layout
            // everything below the panel every few frames. A pass that
            // skipped this frame just holds its smoothed value. gpu_frame_ms
            // is the buffer's wall span (includes the vsync drawable stall);
            // encoder timestamps aren't — the remainder row is idle wait
            // plus the tiny unsampled readback blits.
            static std::vector<std::pair<std::string, float>> pass_avg;
            double pass_sum = 0.0;
            for (const auto& [label, ms] : backend.gpu_pass_ms()) {
                pass_sum += ms;
                // The renderers displace each other: without this, switching
                // modes would leave a stale row for the inactive one forever.
                const char* other = label == "pathtrace" ? "raymarch"
                                  : label == "raymarch"  ? "pathtrace" : nullptr;
                if (other) {
                    std::erase_if(pass_avg, [&](const auto& e) { return e.first == other; });
                }
                auto it = std::find_if(pass_avg.begin(), pass_avg.end(),
                                       [&](const auto& e) { return e.first == label; });
                if (it == pass_avg.end()) it = pass_avg.insert(pass_avg.end(), {label, (float)ms});
                it->second += ((float)ms - it->second) * 0.05f;
            }
            for (const auto& [label, ms] : pass_avg) {
                ImGui::Text("  %-11s %5.2f ms", label.c_str(), ms);
            }
            if (!pass_avg.empty()) {  // empty = sampling unsupported, no breakdown at all
                static float other_avg = 0;
                if (pass_sum > 0.0) other_avg += ((float)(gpu_ms - pass_sum) - other_avg) * 0.05f;
                ImGui::Text("  %-11s %5.2f ms", "idle/vsync", std::max(other_avg, 0.0f));
            }
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
        ImGui::SameLine();
        ImGui::Checkbox("Compact", &g_compact_param_labels);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Full-width sliders with the name drawn over the bar");
        }
        ImGui::SameLine();
        ImGui::BeginDisabled(job.active);
        if (ImGui::Checkbox("Timeline", &timeline.enabled) && !timeline.enabled) {
            timeline.playing = false;
        }
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered()) {
            if (timeline.enabled) {
                ImGui::SetTooltip("Off: the panel hides and keys stop driving anything.\n"
                                  "Nothing is deleted. Space plays/pauses.");
            } else {
                ImGui::SetTooltip("%zu param track(s), %zu camera key(s) parked",
                                  timeline.tracks.size(), timeline.cam_keys.size());
            }
        }
        render_shader_errors(shaders);

        if (ImGui::CollapsingHeader("Camera")) {
            const char* mode_names[] = {"Trackball", "FPS"};
            int mode_idx = (int)camera.mode;
            if (ImGui::Combo("Mode", &mode_idx, mode_names, IM_ARRAYSIZE(mode_names))) {
                camera.mode = (CameraMode)mode_idx;
            }
            // Typed pose entry — exact, repeatable camera keys: dial in
            // numbers, then Key Camera (or let auto-key catch the edit).
            // Disabled while the timeline owns the camera, same as the mouse.
            // Values apply live per keystroke; auto-key fires once, when a
            // field loses focus after an edit.
            ImGui::BeginDisabled(cam_locked);
            bool cam_typed = false;
            if (ImGui::Button("Reset Camera")) {
                camera.pos[0] = 0; camera.pos[1] = 0; camera.pos[2] = 1;
                camera.target[0] = 0; camera.target[1] = 0; camera.target[2] = 0;
            }
            ImGui::InputFloat3("Target", camera.target, "%.7g");
            cam_typed |= ImGui::IsItemDeactivatedAfterEdit();
            ImGui::InputFloat3("Position", camera.pos, "%.7g");
            cam_typed |= ImGui::IsItemDeactivatedAfterEdit();
            {
                float off[3];
                v3::sub(camera.pos, camera.target, off);
                float cur = v3::length(off);
                float dist = cur;
                // Moves pos along the current view direction. Live-typed
                // intermediates like the "0" in "0.001" are skipped so a
                // transient zero can't collapse pos onto target and lose the
                // direction — and deliberately not clamped to the trackball's
                // zoom limits, typing exact tiny distances is the point.
                if (ImGui::InputFloat("Distance", &dist, 0, 0, "%.7g") &&
                    dist > 0.0f && cur > 0.0f) {
                    v3::mad(camera.target, off, dist / cur, camera.pos);
                }
                cam_typed |= ImGui::IsItemDeactivatedAfterEdit();
            }
            ImGui::SliderAngle("FOV", &camera.fov, 5.0f, 160.0f);
            cam_typed |= ImGui::IsItemDeactivatedAfterEdit();
            ImGui::EndDisabled();
            if (cam_locked) ImGui::TextDisabled("(timeline is driving the camera)");

            // Typed edits sequence exactly like mouse moves do.
            if (cam_typed && timeline.enabled && timeline.auto_key && timeline.drive_camera) {
                timeline.key_camera(timeline.playhead, camera);
            }

            ImGui::Text("Fwd:    %.3f, %.3f, %.3f", fwd[0], fwd[1], fwd[2]);
            ImGui::Text("Up:     %.3f, %.3f, %.3f", up[0], up[1], up[2]);
            ImGui::Text("Right:  %.3f, %.3f, %.3f", right[0], right[1], right[2]);
            ImGui::Text("Surface Dist: %.4f", camera.nav_distance);
            ImGui::Checkbox("Adaptive Speed", &camera.adaptive_speed);
            ImGui::SliderFloat("Rotate Speed", &camera.rotate_speed, 0.01f, 5.0f);
            ImGui::SliderFloat("Pan Speed", &camera.pan_speed, 0.01f, 5.0f);
            ImGui::SliderFloat("Zoom Speed", &camera.zoom_speed, 0.01f, 1.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("Keyboard Speed", &camera.keyboard_speed, 0.01f, 2.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
        }

        // A preset carries the animation as well as the parameters, so Load,
        // Update and Delete are all destructive to authored work. Clicks only
        // record intent here; the modals below do the actual work.
        PendingPreset::Kind clicked = PendingPreset::None;
        // Outside the header so the overwrite dialog can clear it on confirm.
        static char preset_name[64] = "";

        if (ImGui::CollapsingHeader("Presets")) {
            if (current_preset.empty()) {
                ImGui::TextDisabled("Preset: (none)");
            } else {
                ImGui::Text("Preset: %s%s", current_preset.c_str(), anim_dirty() ? " *" : "");
                if (anim_dirty() && ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Animation edited since load — Update to keep it");
                }
            }

            ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 60);
            ImGui::InputTextWithHint("##preset_name", "preset name", preset_name, sizeof(preset_name));
            ImGui::SameLine();
            if (ImGui::Button("Save") && preset_name[0]) {
                // Typing the name of an existing preset is an overwrite, and
                // deserves the same confirmation as the Update button.
                if (std::find(presets.begin(), presets.end(), preset_name) != presets.end()) {
                    pending.name = preset_name;
                    clicked = PendingPreset::Update;
                } else {
                    save_preset(preset_name);
                    preset_name[0] = 0;
                    refresh_presets();
                }
            }

            for (const auto& p : presets) {
                ImGui::PushID(p.c_str());
                if (ImGui::Button("Load")) {
                    if (anim_dirty()) {
                        pending.name = p;
                        clicked = PendingPreset::Load;
                    } else {
                        load_preset(p);   // nothing to lose, skip the dialog
                    }
                }
                ImGui::SameLine();
                if (ImGui::Button("Update")) {
                    pending.name = p;
                    clicked = PendingPreset::Update;
                }
                ImGui::SameLine();
                if (ImGui::Button("X")) {
                    pending.name = p;
                    clicked = PendingPreset::Delete;
                }
                ImGui::SameLine();
                bool is_current = (p == current_preset);
                if (is_current) ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.45f, 0.85f, 1.0f, 1.0f));
                ImGui::TextUnformatted(p.c_str());
                if (is_current) ImGui::PopStyleColor();
                ImGui::PopID();
            }
        }

        // Modals live outside the CollapsingHeader so an open dialog survives
        // the header being collapsed, and outside the per-row PushID so the
        // OpenPopup/BeginPopupModal IDs match. OpenPopup fires only on the
        // click frame — calling it every frame would make Esc unable to close.
        if (clicked != PendingPreset::None) {
            pending.kind = clicked;
            ImGui::OpenPopup(preset_popup_title(clicked));
        }

        if (ImGui::BeginPopupModal(preset_popup_title(PendingPreset::Load), nullptr,
                                   ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("%s has unsaved animation changes\n(%zu camera keys, %zu param tracks).",
                        current_preset.empty() ? "The current animation" : current_preset.c_str(),
                        timeline.cam_keys.size(), timeline.tracks.size());
            ImGui::Text("Loading \"%s\" will replace them.", pending.name.c_str());
            ImGui::Separator();
            if (!current_preset.empty()) {
                char lbl[96];
                std::snprintf(lbl, sizeof(lbl), "Save to %s", current_preset.c_str());
                if (ImGui::Button(lbl)) {
                    save_preset(current_preset);
                    load_preset(pending.name);
                    ImGui::CloseCurrentPopup();
                }
                ImGui::SameLine();
            }
            if (ImGui::Button("Discard")) {
                load_preset(pending.name);
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel")) ImGui::CloseCurrentPopup();
            ImGui::EndPopup();
        }

        if (ImGui::BeginPopupModal(preset_popup_title(PendingPreset::Update), nullptr,
                                   ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("Overwrite \"%s\"?", pending.name.c_str());
            ImGui::TextDisabled("Replaces its parameters, camera, ranges and animation.");
            ImGui::Separator();
            if (ImGui::Button("Overwrite")) {
                save_preset(pending.name);
                // Clears the name box when the overwrite came from Save.
                if (pending.name == preset_name) preset_name[0] = 0;
                refresh_presets();
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel")) ImGui::CloseCurrentPopup();
            ImGui::EndPopup();
        }

        if (ImGui::BeginPopupModal(preset_popup_title(PendingPreset::Delete), nullptr,
                                   ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("Delete \"%s\"?", pending.name.c_str());
            ImGui::TextDisabled("Its animation goes with it. This cannot be undone.");
            ImGui::Separator();
            if (ImGui::Button("Delete")) {
                std::error_code ec;
                std::filesystem::remove(preset_dir + pending.name + ".json", ec);
                if (ec) std::cerr << "[preset] delete failed: " << ec.message() << "\n";
                // The state stays loaded; it just no longer belongs to a file.
                if (current_preset == pending.name) current_preset.clear();
                refresh_presets();
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel")) ImGui::CloseCurrentPopup();
            ImGui::EndPopup();
        }

        // Esc closes a modal without running any branch above — drop the
        // intent so a stale name can't be picked up by the next dialog.
        if (pending.kind != PendingPreset::None &&
            !ImGui::IsPopupOpen(preset_popup_title(pending.kind))) {
            pending = PendingPreset{};
        }

        // Host state that shader `@if` clauses can test by name
        ParamContext param_ctx = {
            {"renderer", renderer_mode == 1 ? "pathtrace" : "raymarch"},
        };

        // nullptr when the timeline is off: no tint, no keying context menu,
        // no auto-key — the sliders behave exactly as they did before it existed.
        Timeline* tl = timeline.enabled ? &timeline : nullptr;

        if (ImGui::CollapsingHeader("Shader Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
            render_shader_params(shaders.get_params_mut("raymarch.metal"), param_ctx,
                                 tl, "raymarch.metal");
        }
        if (renderer_mode == 1 && ImGui::CollapsingHeader("Path Tracer", ImGuiTreeNodeFlags_DefaultOpen)) {
            render_shader_params(shaders.get_params_mut("pathtrace.metal"), param_ctx,
                                 tl, "pathtrace.metal");
        }
        if (ImGui::CollapsingHeader("Reconstruction", ImGuiTreeNodeFlags_DefaultOpen)) {
            render_shader_params(shaders.get_params_mut("reconstruct.metal"), param_ctx,
                                 tl, "reconstruct.metal");
        }
        if (ImGui::CollapsingHeader("Post", ImGuiTreeNodeFlags_DefaultOpen)) {
            render_shader_params(shaders.get_params_mut("present.metal"), param_ctx,
                                 tl, "present.metal");
        }
        ImGui::End();

        // --- Timeline ---
        // Passing &timeline.enabled gives the window a close button, so the X
        // and the checkbox in the main panel are the same switch.
        if (timeline.enabled) {
        // No close button mid-render — the Cancel button lives in this window.
        ImGui::Begin("Timeline", job.active ? nullptr : &timeline.enabled);
        draw_timeline(timeline, timeline_ui, shaders, camera);

        ImGui::Spacing();
        // Presets own the current animation now; this library is for reusing
        // one across presets, not for storing the live one.
        if (ImGui::CollapsingHeader("Import / Export")) {
            static char anim_name[64] = "";
            ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 60);
            ImGui::InputTextWithHint("##anim_name", "animation name", anim_name, sizeof(anim_name));
            ImGui::SameLine();
            if (ImGui::Button("Save") && anim_name[0]) {
                timeline.save(anim_dir + anim_name + ".json");
                anim_name[0] = 0;
                refresh_anims();
            }
            std::string anim_delete;
            for (const auto& a : anims) {
                if (a == "_working") continue;
                ImGui::PushID(a.c_str());
                if (ImGui::Button("Load")) {
                    timeline.load(anim_dir + a + ".json");
                    timeline_ui.view_start = 0.0f;
                    timeline_ui.view_end = timeline.duration;
                }
                ImGui::SameLine();
                if (ImGui::Button("Update")) timeline.save(anim_dir + a + ".json");
                ImGui::SameLine();
                if (ImGui::Button("X") && ImGui::GetIO().KeyShift) anim_delete = a;
                if (ImGui::IsItemHovered() && !ImGui::GetIO().KeyShift) {
                    ImGui::SetTooltip("shift-click to delete");
                }
                ImGui::SameLine();
                ImGui::TextUnformatted(a.c_str());
                ImGui::PopID();
            }
            if (!anim_delete.empty()) {
                std::error_code ec;
                std::filesystem::remove(anim_dir + anim_delete + ".json", ec);
                refresh_anims();
            }
        }

        // --- Offline render ---
        if (ImGui::CollapsingHeader("Render", ImGuiTreeNodeFlags_DefaultOpen)) {
            static char out_dir[256] = "";
            if (out_dir[0] == 0) {
                std::snprintf(out_dir, sizeof(out_dir), "%srenders/", SDL_GetBasePath());
            }
            if (s_picked_ready.load(std::memory_order_acquire)) {
                std::snprintf(out_dir, sizeof(out_dir), "%s/", s_picked_dir);
                s_picked_ready.store(false, std::memory_order_relaxed);
            }
            ImGui::BeginDisabled(job.active);
            ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 40);
            ImGui::InputTextWithHint("##outdir", "output directory", out_dir, sizeof(out_dir));
            ImGui::SameLine();
            if (ImGui::Button("...", ImVec2(32, 0))) {
                SDL_ShowOpenFolderDialog(on_render_dir_picked, nullptr, window,
                                         out_dir[0] ? out_dir : nullptr, false);
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Choose the render output folder");

            static int range[2] = {0, -1};
            if (range[1] < 0) range[1] = (int)std::lround(timeline.duration * timeline.fps);
            ImGui::DragInt2("Frame range", range, 1.0f, 0, 1000000);
            int samples = (int)job.samples;
            if (ImGui::DragInt("Samples/frame", &samples, 1.0f, 1, 16384)) {
                job.samples = (uint32_t)std::max(samples, 1);
            }
            ImGui::SliderFloat("Shutter", &job.shutter, 0.0f, 1.0f, "%.2f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Fraction of a frame the shutter is open.\n"
                                  "0 = static poses (no motion blur).");
            }
            ImGui::SliderFloat("Render scale", &job.scale, 0.25f, 4.0f, "%.2fx");
            ImGui::Checkbox("16-bit PNG", &job.png16);
            ImGui::EndDisabled();

            if (!job.active) {
                if (ImGui::Button("Start Render", ImVec2(120, 0))) {
                    job.dir = out_dir;
                    if (!job.dir.empty() && job.dir.back() != '/') job.dir += '/';
                    std::error_code ec;
                    std::filesystem::create_directories(job.dir, ec);
                    if (ec) {
                        std::cerr << "[render] cannot create " << job.dir << ": "
                                  << ec.message() << "\n";
                    } else {
                        // Provenance beside the frames: the exact state and
                        // animation this render came from. state.json is a
                        // loadable preset (timeline embedded); animation.json
                        // imports from the animations library.
                        state.save_to(job.dir + "state.json", shaders, &timeline);
                        timeline.save(job.dir + "animation.json");

                        job.frame_first = std::min(range[0], range[1]);
                        job.frame_last  = std::max(range[0], range[1]);
                        job.frame = job.frame_first;
                        job.sample = 0;
                        job.needs_reset = true;
                        job.failures = 0;
                        job.avg_frame_secs = 0;
                        job.started_at = time;
                        job.frame_started_at = time;
                        // Even dimensions: the trace pass runs at exactly half
                        // resolution and an odd size would drop a column.
                        job_w = (uint32_t)std::max(2.0f, backend.drawable_width() * job.scale) & ~1u;
                        job_h = (uint32_t)std::max(2.0f, backend.drawable_height() * job.scale) & ~1u;
                        timeline.playing = false;
                        job.active = true;
                        backend.set_vsync(false);
                    }
                }
                ImGui::SameLine();
                ImGui::TextDisabled("%d frames @ %u spp", std::abs(range[1] - range[0]) + 1,
                                    job.samples);
            } else {
                char pb[96];
                char eta[24] = "--";  // no estimate until the first frame lands
                if (job.avg_frame_secs > 0.0) {
                    format_duration(job.avg_frame_secs * (job.total_frames() - job.done_frames()),
                                    eta, sizeof(eta));
                }
                std::snprintf(pb, sizeof(pb), "frame %d/%d  sample %u/%u  ETA %s",
                              job.done_frames() + 1, job.total_frames(),
                              job.sample, job.samples, eta);
                ImGui::ProgressBar(job.progress(), ImVec2(-1, 0), pb);
                if (ImGui::Button("Cancel", ImVec2(120, 0))) end_job();
                ImGui::SameLine();
                ImGui::TextDisabled("(or press Esc)");
            }
        }
        ImGui::End();
        }  // timeline.enabled

        ImGui::Render();
        backend.render_imgui();

        backend.end_frame();

        // --- Write the captured frame ---
        // Blocking on the GPU is correct here: determinism beats throughput,
        // and it happens once per output frame, not once per sample.
        if (job.capture_pending) {
            job.capture_pending = false;
            backend.wait_last_frame();

            char name[64];
            std::snprintf(name, sizeof(name), "frame_%05d.png", job.frame);
            if (!save_rgba16f_png(backend.buffer_contents(buf_capture), w, h, capture_row,
                                  job.png16, job.dir + name)) {
                std::cerr << "[render] failed to write " << job.dir << name << "\n";
                if (++job.failures >= 3) {
                    std::cerr << "[render] aborting after 3 write failures\n";
                    end_job();
                }
            }

            if (job.active) {
                double now = (double)(SDL_GetPerformanceCounter() - start_time) / (double)freq;
                double secs = now - job.frame_started_at;
                job.avg_frame_secs = job.avg_frame_secs > 0 ? job.avg_frame_secs * 0.7 + secs * 0.3
                                                            : secs;
                job.frame_started_at = now;
                job.sample = 0;
                job.needs_reset = true;   // a new pose is a genuine scene change
                if (++job.frame > job.frame_last) {
                    std::cout << "[render] finished " << job.total_frames() << " frames -> "
                              << job.dir << "\n";
                    end_job();
                }
            }
        }

        // Auto-save state if anything changed. Suspended while the timeline is
        // driving values: animated params would churn state.json and overwrite
        // the authored static values with wherever the playhead happened to be.
        if (!job.active && !timeline.playing) {
            state.save_if_changed(shaders, time);
        }

        // Working-animation autosave, same 2s debounce as StateSerializer.
        if (timeline.revision != anim_saved_revision) {
            if (anim_dirty_since < 0) anim_dirty_since = time;
            if (time - anim_dirty_since > 2.0f) {
                if (timeline.save(anim_autosave)) anim_saved_revision = timeline.revision;
                anim_dirty_since = -1.0f;
            }
        } else {
            anim_dirty_since = -1.0f;
        }

        frame_index++;
    }

    // Cleanup — force a save so debounced changes survive quit/restart
    state.save(shaders);
    if (timeline.revision != anim_saved_revision) timeline.save(anim_autosave);
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
