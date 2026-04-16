#include "gui.h"
#include "math_util.h"
#include "imgui.h"
#include <cmath>
#include <cstring>

// ---- Camera ----

void Camera::update(float dt, const SDL_Event* events, int event_count) {
    // Keyboard movement
    const bool* keys = SDL_GetKeyboardState(nullptr);
    float fwd[3], up[3], right[3];
    get_vectors(fwd, up, right);

    float move_speed = speed * dt;

    if (keys[SDL_SCANCODE_W]) { pos[0] += fwd[0]*move_speed; pos[1] += fwd[1]*move_speed; pos[2] += fwd[2]*move_speed; }
    if (keys[SDL_SCANCODE_S]) { pos[0] -= fwd[0]*move_speed; pos[1] -= fwd[1]*move_speed; pos[2] -= fwd[2]*move_speed; }
    if (keys[SDL_SCANCODE_D]) { pos[0] += right[0]*move_speed; pos[1] += right[1]*move_speed; pos[2] += right[2]*move_speed; }
    if (keys[SDL_SCANCODE_A]) { pos[0] -= right[0]*move_speed; pos[1] -= right[1]*move_speed; pos[2] -= right[2]*move_speed; }
    if (keys[SDL_SCANCODE_E] || keys[SDL_SCANCODE_SPACE]) { pos[1] += move_speed; }
    if (keys[SDL_SCANCODE_Q] || keys[SDL_SCANCODE_LSHIFT]) { pos[1] -= move_speed; }

    // Mouse look (right-click drag)
    for (int i = 0; i < event_count; i++) {
        const auto& e = events[i];
        if (e.type == SDL_EVENT_MOUSE_MOTION && (e.motion.state & SDL_BUTTON_RMASK)) {
            yaw   += e.motion.xrel * sensitivity;
            pitch -= e.motion.yrel * sensitivity;
            // Clamp pitch
            if (pitch > 1.5f) pitch = 1.5f;
            if (pitch < -1.5f) pitch = -1.5f;
        }
        if (e.type == SDL_EVENT_MOUSE_WHEEL) {
            speed *= (e.wheel.y > 0) ? 1.2f : (1.0f / 1.2f);
            if (speed < 0.01f) speed = 0.01f;
            if (speed > 100.0f) speed = 100.0f;
        }
    }
}

void Camera::get_vectors(float* fwd, float* up, float* right) const {
    fwd[0] = std::cos(pitch) * std::sin(yaw);
    fwd[1] = std::sin(pitch);
    fwd[2] = std::cos(pitch) * std::cos(yaw);

    // World up
    float world_up[3] = {0, 1, 0};

    // right = normalize(cross(fwd, world_up))
    right[0] = fwd[1] * world_up[2] - fwd[2] * world_up[1];
    right[1] = fwd[2] * world_up[0] - fwd[0] * world_up[2];
    right[2] = fwd[0] * world_up[1] - fwd[1] * world_up[0];
    float rl = std::sqrt(right[0]*right[0] + right[1]*right[1] + right[2]*right[2]);
    if (rl > 1e-6f) { right[0] /= rl; right[1] /= rl; right[2] /= rl; }

    // up = cross(right, fwd)
    up[0] = right[1] * fwd[2] - right[2] * fwd[1];
    up[1] = right[2] * fwd[0] - right[0] * fwd[2];
    up[2] = right[0] * fwd[1] - right[1] * fwd[0];
}

void Camera::get_view_matrix(float* out_4x4) const {
    float fwd[3], up[3], right[3];
    get_vectors(fwd, up, right);
    float center[3] = { pos[0] + fwd[0], pos[1] + fwd[1], pos[2] + fwd[2] };
    mat4::look_at(pos, center, up, out_4x4);
}

void Camera::get_projection_matrix(float aspect, float near_z, float far_z, float* out_4x4) const {
    mat4::perspective(fov, aspect, near_z, far_z, out_4x4);
}

void Camera::get_view_proj(float aspect, float near_z, float far_z, float* out_4x4) const {
    float view[16], proj[16];
    get_view_matrix(view);
    get_projection_matrix(aspect, near_z, far_z, proj);
    mat4::multiply(proj, view, out_4x4);
}

// ---- ImGui helpers ----

void render_shader_params(std::vector<ShaderParam>& params) {
    for (auto& p : params) {
        switch (p.type) {
            case ShaderParam::Float:
                ImGui::SliderFloat(p.name.c_str(), &p.current_val[0],
                                   p.min_val[0], p.max_val[0]);
                break;
            case ShaderParam::Int: {
                int v = (int)p.current_val[0];
                if (ImGui::SliderInt(p.name.c_str(), &v,
                                     (int)p.min_val[0], (int)p.max_val[0])) {
                    p.current_val[0] = (float)v;
                }
                break;
            }
            case ShaderParam::Float2:
                ImGui::SliderFloat2(p.name.c_str(), p.current_val,
                                    p.min_val[0], p.max_val[0]);
                break;
            case ShaderParam::Float3:
                if (p.is_color) {
                    ImGui::ColorEdit3(p.name.c_str(), p.current_val);
                } else {
                    ImGui::SliderFloat3(p.name.c_str(), p.current_val,
                                        p.min_val[0], p.max_val[0]);
                }
                break;
            case ShaderParam::Float4:
                if (p.is_color) {
                    ImGui::ColorEdit4(p.name.c_str(), p.current_val);
                } else {
                    ImGui::SliderFloat4(p.name.c_str(), p.current_val,
                                        p.min_val[0], p.max_val[0]);
                }
                break;
        }
    }
}

void render_shader_errors(const ShaderManager& sm) {
    for (const auto& entry : sm.entries()) {
        if (!entry.error.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, 0.3f, 0.3f, 1));
            ImGui::TextWrapped("[%s] %s", entry.filename.c_str(), entry.error.c_str());
            ImGui::PopStyleColor();
        }
    }
}
