#pragma once
#include "types.h"
#include "shader_manager.h"
#include <SDL3/SDL.h>
#include <string>
#include <utility>
#include <vector>

enum class CameraMode { Trackball, FPS };

// Camera supports two modes:
//   Trackball: left-drag orbits target, right-drag pans, wheel zooms.
//   FPS:       left-drag rotates view around pos; right-drag and wheel are no-ops.
// pos/target are the source of truth in both modes; mode switch preserves view direction.
struct Camera {
    CameraMode mode    = CameraMode::Trackball;
    float pos[3]       = {3, 3, -5};
    float target[3]    = {0, 0, 0};
    float fov          = 1.2f;     // radians (~70 degrees)
    bool  show_grid    = true;

    // Controller settings
    float rotate_speed   = 1.0f;
    float pan_speed      = 1.0f;
    float zoom_speed     = 0.2f;
    float keyboard_speed = 0.5f;
    float min_distance   = 0.01f;
    float max_distance   = 1000.0f;

    // Scale keyboard speed by distance to the fractal surface (depth readback)
    bool  adaptive_speed = true;
    // Smoothed distance to fractal surface (from depth readback); <= 0 until first readback.
    float nav_distance   = -1.0f;

    // Derived vectors (computed by get_vectors)
    void get_vectors(float* fwd, float* up, float* right) const;
    void get_view_matrix(float* out_4x4) const;
    void get_projection_matrix(float aspect, float near_z, float far_z, float* out_4x4) const;
    void get_view_proj(float aspect, float near_z, float far_z, float* out_4x4) const;

    // Input handling
    void update(float dt, const SDL_Event* events, int event_count,
                float screen_w, float screen_h);

private:
    bool rotating_ = false;
    bool panning_  = false;
    float last_mouse_[2] = {};

    void rotate(float dx, float dy, float screen_w, float screen_h);
    void look(float dx, float dy);
    void pan(float dx, float dy, float screen_w, float screen_h);
    void zoom(float delta);
    void move_keyboard(float dt);
};

// ImGui helpers

// Host-side values a param's `@if` clause can test by name, for state the
// shader can't see (e.g. {"renderer", "pathtrace"}).
using ParamContext = std::vector<std::pair<std::string, std::string>>;

struct Timeline;  // timeline.h — forward declared to keep the include one-way

// Passing `tl` + `shader_file` turns every param in the list into an
// animatable one: right-click for keying, animated params are tinted, and
// drags feed auto-key. Because every `@param` in the engine flows through this
// one function, that covers params that don't exist yet.
// Compact layout: sliders span the full panel width and the param name is
// painted over the bar instead of sitting beside it. ImGui's default puts the
// label to the right of every widget, which on a 23-param panel costs more
// horizontal space than the sliders themselves.
extern bool g_compact_param_labels;

// Neutralises the dark theme's blue slider track and grab to gray. Call once
// after ImGui::StyleColorsDark().
void apply_gui_style();

void render_shader_params(std::vector<ShaderParam>& params, const ParamContext& ctx = {},
                          Timeline* tl = nullptr, const char* shader_file = nullptr);
void render_shader_errors(const ShaderManager& sm);
