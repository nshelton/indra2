#pragma once
#include "types.h"
#include "shader_manager.h"
#include <SDL3/SDL.h>
#include <vector>

// Trackball camera (orbits around a target point)
struct Camera {
    float pos[3]       = {3, 3, -5};
    float target[3]    = {0, 0, 0};
    float fov          = 1.2f;     // radians (~70 degrees)
    bool  show_grid    = true;

    // Controller settings
    float rotate_speed   = 1.0f;
    float pan_speed      = 1.0f;
    float zoom_speed     = 0.3f;
    float keyboard_speed = 1.0f;
    float min_distance   = 0.01f;
    float max_distance   = 1000.0f;

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
    void pan(float dx, float dy, float screen_w, float screen_h);
    void zoom(float delta);
    void move_keyboard(float dt);
};

// ImGui helpers
void render_shader_params(std::vector<ShaderParam>& params);
void render_shader_errors(const ShaderManager& sm);
