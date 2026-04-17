#include "gui.h"
#include "math_util.h"
#include "imgui.h"
#include <cmath>
#include <cstring>
#include <algorithm>

// ---- Camera ----

void Camera::get_vectors(float* fwd, float* up, float* right) const {
    float dir[3];
    v3::sub(target, pos, dir);
    v3::normalize(dir, fwd);

    float world_up[3] = {0, 1, 0};
    v3::cross(fwd, world_up, right);
    float rl = v3::length(right);
    if (rl > 1e-6f) {
        right[0] /= rl; right[1] /= rl; right[2] /= rl;
    } else {
        right[0] = 1; right[1] = 0; right[2] = 0;
    }

    v3::cross(right, fwd, up);
}

void Camera::get_view_matrix(float* out_4x4) const {
    float up[3] = {0, 1, 0};
    mat4::look_at(pos, target, up, out_4x4);
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

// ---- Trackball controls ----

void Camera::update(float dt, const SDL_Event* events, int event_count,
                    float screen_w, float screen_h) {
    for (int i = 0; i < event_count; i++) {
        const auto& e = events[i];

        if (e.type == SDL_EVENT_MOUSE_BUTTON_DOWN || e.type == SDL_EVENT_MOUSE_BUTTON_UP) {
            if (ImGui::GetIO().WantCaptureMouse) continue;
            bool down = (e.type == SDL_EVENT_MOUSE_BUTTON_DOWN);
            if (e.button.button == SDL_BUTTON_LEFT)  rotating_ = down;
            if (e.button.button == SDL_BUTTON_RIGHT) panning_  = down;
            if (down) {
                last_mouse_[0] = e.button.x;
                last_mouse_[1] = e.button.y;
            }
        }

        if (e.type == SDL_EVENT_MOUSE_MOTION) {
            if (ImGui::GetIO().WantCaptureMouse) continue;
            float dx = e.motion.x - last_mouse_[0];
            float dy = e.motion.y - last_mouse_[1];
            last_mouse_[0] = e.motion.x;
            last_mouse_[1] = e.motion.y;

            if (rotating_) rotate(dx, dy, screen_w, screen_h);
            if (panning_)  pan(dx, dy, screen_w, screen_h);
        }

        if (e.type == SDL_EVENT_MOUSE_WHEEL) {
            if (ImGui::GetIO().WantCaptureMouse) continue;
            zoom(e.wheel.y);
        }
    }

    if (!ImGui::GetIO().WantCaptureKeyboard) {
        move_keyboard(dt);
    }
}

void Camera::rotate(float dx, float dy, float screen_w, float screen_h) {
    if (dx == 0 && dy == 0) return;

    float offset[3];
    v3::sub(pos, target, offset);

    // Horizontal rotation around world Y
    float angleX = -dx * rotate_speed * 0.005f;
    float y_axis[3] = {0, 1, 0};
    float q_y[4];
    quat::from_axis_angle(y_axis, angleX, q_y);

    float offset_after_y[3];
    quat::rotate_vec3(q_y, offset, offset_after_y);

    // Vertical rotation: compute right vector, clamp elevation
    float fwd_dir[3];
    v3::scale(offset_after_y, -1.0f, fwd_dir);
    v3::normalize(fwd_dir, fwd_dir);

    float world_up[3] = {0, 1, 0};
    float right_vec[3];
    v3::cross(fwd_dir, world_up, right_vec);
    float rl = v3::length(right_vec);
    if (rl < 0.01f) {
        // Near pole fallback
        right_vec[0] = -offset_after_y[2];
        right_vec[1] = 0;
        right_vec[2] = offset_after_y[0];
        rl = v3::length(right_vec);
        if (rl < 0.01f) { right_vec[0] = 1; right_vec[1] = 0; right_vec[2] = 0; rl = 1; }
    }
    v3::scale(right_vec, 1.0f / rl, right_vec);

    // Clamp vertical angle to prevent gimbal lock
    float horiz[3] = {offset_after_y[0], 0, offset_after_y[2]};
    float horiz_dist = v3::length(horiz);
    float current_elev = std::atan2(offset_after_y[1], horiz_dist);
    float angleY = -dy * rotate_speed * 0.005f;
    float new_elev = std::clamp(current_elev + angleY, -1.4f, 1.4f);
    angleY = new_elev - current_elev;

    float q_x[4];
    quat::from_axis_angle(right_vec, angleY, q_x);

    float new_offset[3];
    quat::rotate_vec3(q_x, offset_after_y, new_offset);

    v3::add(target, new_offset, pos);
}

void Camera::pan(float dx, float dy, float screen_w, float screen_h) {
    if (dx == 0 && dy == 0) return;

    float offset[3];
    v3::sub(pos, target, offset);
    float distance = v3::length(offset);

    float pan_scale = distance * pan_speed * 0.001f;

    float fwd[3], up[3], right[3];
    get_vectors(fwd, up, right);

    float world_up[3] = {0, 1, 0};
    float pan_offset[3];
    v3::scale(right, -dx * pan_scale, pan_offset);
    v3::mad(pan_offset, world_up, dy * pan_scale, pan_offset);

    v3::add(pos, pan_offset, pos);
    v3::add(target, pan_offset, target);
}

void Camera::zoom(float delta) {
    if (delta == 0) return;

    float offset[3];
    v3::sub(pos, target, offset);
    float distance = v3::length(offset);

    float zoom_amount = delta * zoom_speed * distance * 0.05f;
    float new_distance = std::clamp(distance - zoom_amount, min_distance, max_distance);

    float dir[3];
    v3::normalize(offset, dir);
    v3::scale(dir, new_distance, offset);
    v3::add(target, offset, pos);
}

void Camera::move_keyboard(float dt) {
    bool w = ImGui::IsKeyDown(ImGuiKey_W);
    bool s = ImGui::IsKeyDown(ImGuiKey_S);
    bool a = ImGui::IsKeyDown(ImGuiKey_A);
    bool d = ImGui::IsKeyDown(ImGuiKey_D);
    bool q = ImGui::IsKeyDown(ImGuiKey_Q);
    bool e = ImGui::IsKeyDown(ImGuiKey_E);

    if (!w && !s && !a && !d && !q && !e) return;

    float fwd[3], up[3], right[3];
    get_vectors(fwd, up, right);

    float move[3] = {0, 0, 0};
    if (w) v3::add(move, fwd, move);
    if (s) { float neg[3]; v3::scale(fwd, -1, neg); v3::add(move, neg, move); }
    if (d) v3::add(move, right, move);
    if (a) { float neg[3]; v3::scale(right, -1, neg); v3::add(move, neg, move); }
    float world_up[3] = {0, 1, 0};
    float world_down[3] = {0, -1, 0};
    if (e) v3::add(move, world_up, move);
    if (q) v3::add(move, world_down, move);

    float ml = v3::length(move);
    if (ml < 0.001f) return;

    // Scale speed by distance to target
    float offset[3];
    v3::sub(pos, target, offset);
    float distance = v3::length(offset);
    float scaled_speed = keyboard_speed * distance * dt;

    v3::normalize(move, move);
    v3::scale(move, scaled_speed, move);

    v3::add(pos, move, pos);
    v3::add(target, move, target);
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
