#pragma once
#include "types.h"
#include "shader_manager.h"
#include <SDL3/SDL.h>
#include <vector>

// FPS camera
struct Camera {
    float pos[3]       = {0, 3, -5};
    float yaw          = 0;        // radians
    float pitch        = -0.5f;    // radians, looking slightly down
    float fov          = 1.2f;     // radians (~70 degrees)
    float speed        = 2.0f;
    float sensitivity  = 0.003f;
    bool  show_grid    = true;

    void update(float dt, const SDL_Event* events, int event_count);
    void get_vectors(float* fwd, float* up, float* right) const;
    void get_view_matrix(float* out_4x4) const;
    void get_projection_matrix(float aspect, float near_z, float far_z, float* out_4x4) const;
    void get_view_proj(float aspect, float near_z, float far_z, float* out_4x4) const;
};

// ImGui helpers
void render_shader_params(std::vector<ShaderParam>& params);
void render_shader_errors(const ShaderManager& sm);
