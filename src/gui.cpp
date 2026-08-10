#include "gui.h"
#include "timeline.h"
#include "math_util.h"
#include "imgui.h"
#include <cfloat>
#include <cmath>
#include <cstdio>
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
            if (e.button.button == SDL_BUTTON_RIGHT && mode == CameraMode::Trackball) panning_ = down;
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

            if (rotating_) {
                if (mode == CameraMode::FPS) look(dx, dy);
                else                         rotate(dx, dy, screen_w, screen_h);
            }
            if (panning_) pan(dx, dy, screen_w, screen_h);
        }

        if (e.type == SDL_EVENT_MOUSE_WHEEL) {
            if (ImGui::GetIO().WantCaptureMouse) continue;
            if (mode == CameraMode::Trackball) zoom(e.wheel.y);
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

    // Scale dy by cos(elev) so vertical input tapers to 0 at the poles —
    // no hard wall, just asymptotic slowdown. The clamp is a numerical
    // safety so we stay off the cross(fwd, world_up) = 0 singularity.
    float horiz[3] = {offset_after_y[0], 0, offset_after_y[2]};
    float horiz_dist = v3::length(horiz);
    float current_elev = std::atan2(offset_after_y[1], horiz_dist);
    float angleY = -dy * rotate_speed * 0.005f * std::cos(current_elev);
    constexpr float MAX_ELEV = 1.555f;
    float new_elev = std::clamp(current_elev + angleY, -MAX_ELEV, MAX_ELEV);
    angleY = new_elev - current_elev;

    float q_x[4];
    quat::from_axis_angle(right_vec, angleY, q_x);

    float new_offset[3];
    quat::rotate_vec3(q_x, offset_after_y, new_offset);

    v3::add(target, new_offset, pos);
}

// FPS look: rotate target around pos. Mirrors rotate() but uses the
// target-relative offset, so the same dx/dy produce the same view rotation.
void Camera::look(float dx, float dy) {
    if (dx == 0 && dy == 0) return;

    float offset[3];
    v3::sub(target, pos, offset);

    float angleX = -dx * rotate_speed * 0.005f;
    float y_axis[3] = {0, 1, 0};
    float q_y[4];
    quat::from_axis_angle(y_axis, angleX, q_y);

    float offset_after_y[3];
    quat::rotate_vec3(q_y, offset, offset_after_y);

    float fwd_dir[3];
    v3::normalize(offset_after_y, fwd_dir);

    float world_up[3] = {0, 1, 0};
    float right_vec[3];
    v3::cross(fwd_dir, world_up, right_vec);
    float rl = v3::length(right_vec);
    if (rl < 0.01f) {
        right_vec[0] = -offset_after_y[2];
        right_vec[1] = 0;
        right_vec[2] = offset_after_y[0];
        rl = v3::length(right_vec);
        if (rl < 0.01f) { right_vec[0] = 1; right_vec[1] = 0; right_vec[2] = 0; rl = 1; }
    }
    v3::scale(right_vec, 1.0f / rl, right_vec);

    float horiz[3] = {offset_after_y[0], 0, offset_after_y[2]};
    float horiz_dist = v3::length(horiz);
    float current_elev = std::atan2(offset_after_y[1], horiz_dist);
    float angleY = -dy * rotate_speed * 0.005f * std::cos(current_elev);
    constexpr float MAX_ELEV = 1.555f;
    float new_elev = std::clamp(current_elev + angleY, -MAX_ELEV, MAX_ELEV);
    angleY = new_elev - current_elev;

    float q_x[4];
    quat::from_axis_angle(right_vec, angleY, q_x);

    float new_offset[3];
    quat::rotate_vec3(q_x, offset_after_y, new_offset);

    v3::add(pos, new_offset, target);
}

void Camera::pan(float dx, float dy, float screen_w, float screen_h) {
    if (dx == 0 && dy == 0) return;

    float offset[3];
    v3::sub(pos, target, offset);
    float distance = v3::length(offset);

    float pan_scale = distance * pan_speed * 0.001f;

    float fwd[3], up[3], right[3];
    get_vectors(fwd, up, right);

    float pan_offset[3];
    v3::scale(right, -dx * pan_scale, pan_offset);
    v3::mad(pan_offset, up, dy * pan_scale, pan_offset);

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
    if (e) v3::add(move, up, move);
    if (q) { float neg[3]; v3::scale(up, -1, neg); v3::add(move, neg, move); }

    float ml = v3::length(move);
    if (ml < 0.001f) return;

    // Scale speed by distance to the fractal surface; fall back to
    // distance-to-target when disabled or before the first depth readback lands.
    float distance = adaptive_speed ? nav_distance : -1.0f;
    if (distance <= 0) {
        float offset[3];
        v3::sub(pos, target, offset);
        distance = v3::length(offset);
    }
    distance = std::max(distance, min_distance);
    float scaled_speed = keyboard_speed * distance * dt;

    v3::normalize(move, move);
    v3::scale(move, scaled_speed, move);

    v3::add(pos, move, pos);
    v3::add(target, move, target);
}

// ---- ImGui helpers ----

// ImGui rounds the stored value to the display format's precision on every
// edit, so the format has to track the range or a drag quantizes the value
// away (lod_factor's 0..0.0002 was stuck at 0 under the default "%.3f").
// Ranges are user-editable now, so derive the precision from the span
// instead of switching on one hardcoded threshold: ~4 significant digits
// across the span, clamped to something printf and a human can both read.
//
// Log sliders lean on this too — SliderBehaviorT sets its zero-crossing floor
// to pow(0.1, precision-parsed-from-format), so too coarse a format collapses
// a small range into the deadzone.
static const char* slider_fmt(const ShaderParam& p) {
    static char buf[8];
    float span = std::fabs(p.max_val[0] - p.min_val[0]);
    int dec = 3;
    if (span > 0.0f) dec = std::clamp(3 - (int)std::floor(std::log10(span)), 2, 9);
    std::snprintf(buf, sizeof(buf), "%%.%df", dec);
    return buf;
}

static ImGuiSliderFlags slider_flags(const ShaderParam& p) {
    return p.logarithmic ? ImGuiSliderFlags_Logarithmic : 0;
}

bool g_compact_param_labels = true;

// ImGui's dark theme tints the slider track and grab blue (FrameBg
// 0.16/0.29/0.48, SliderGrab 0.24/0.52/0.88). Desaturate both to gray, and
// keep the theme's original alphas so they composite over the window
// background exactly as before. FrameBg is shared with combos, inputs and
// checkboxes, so those go gray too — which is the point, they sit in the same
// panel. Accent colors (buttons, headers, the checkbox tick) are left blue.
void apply_gui_style() {
    ImVec4* c = ImGui::GetStyle().Colors;
    c[ImGuiCol_FrameBg]          = ImVec4(0.24f, 0.24f, 0.25f, 0.54f);
    c[ImGuiCol_FrameBgHovered]   = ImVec4(0.42f, 0.42f, 0.44f, 0.40f);
    c[ImGuiCol_FrameBgActive]    = ImVec4(0.50f, 0.50f, 0.52f, 0.67f);
    c[ImGuiCol_SliderGrab]       = ImVec4(0.58f, 0.58f, 0.60f, 1.00f);
    c[ImGuiCol_SliderGrabActive] = ImVec4(0.78f, 0.78f, 0.80f, 1.00f);
}

// NoOptions stops ImGui's own swatch context menu from competing with the
// param popup for the same right-click. In compact mode NoInputs collapses
// the widget to a single square (the RGB fields move into the param popup),
// which is what makes room for the name beside it.
static ImGuiColorEditFlags color_flags(bool compact) {
    return compact ? (ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoOptions)
                   : ImGuiColorEditFlags_NoOptions;
}

// Paints text over the last item, vertically centred and clipped to its rect.
// Pure draw-list work — it submits no item, so it can't disturb the
// LastItemData that IsItemActive()/BeginPopupContextItem() read afterwards.
// The dark copy underneath keeps the name legible where it crosses the
// slider's filled portion.
static void overlay_label(const char* text, ImU32 col) {
    ImVec2 rmin = ImGui::GetItemRectMin();
    ImVec2 rmax = ImGui::GetItemRectMax();
    float pad = ImGui::GetStyle().FramePadding.x + 2.0f;
    float y = rmin.y + (rmax.y - rmin.y - ImGui::GetTextLineHeight()) * 0.5f;
    ImVec2 pos(rmin.x + pad, y);

    ImVec4 clip(rmin.x + 2.0f, rmin.y, rmax.x - 2.0f, rmax.y);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImFont* font = ImGui::GetFont();
    float fs = ImGui::GetFontSize();
    dl->AddText(font, fs, ImVec2(pos.x + 1.0f, pos.y + 1.0f), IM_COL32(0, 0, 0, 170),
                text, nullptr, 0.0f, &clip);
    dl->AddText(font, fs, pos, col, text, nullptr, 0.0f, &clip);
}

// Current value of a `@if` controller, as a string: an enum param in the same
// shader resolves to its selected label, otherwise the host context is
// consulted. nullptr means "unknown", which keeps the dependent param visible
// — a typo in a hot-reloaded shader shouldn't silently hide half the panel.
static const std::string* condition_value(const std::string& name,
                                          const std::vector<ShaderParam>& params,
                                          const ParamContext& ctx) {
    for (const auto& p : params) {
        if (p.name != name) continue;
        if (p.type != ShaderParam::Enum || p.labels.empty()) return nullptr;
        int i = std::clamp((int)p.current_val[0], 0, (int)p.labels.size() - 1);
        return &p.labels[i];
    }
    for (const auto& kv : ctx) {
        if (kv.first == name) return &kv.second;
    }
    return nullptr;
}

static bool param_visible(const ShaderParam& p, const std::vector<ShaderParam>& params,
                          const ParamContext& ctx) {
    for (const auto& c : p.conditions) {
        const std::string* v = condition_value(c.name, params, ctx);
        if (!v) continue;
        bool match = std::find(c.values.begin(), c.values.end(), *v) != c.values.end();
        if (match == c.negate) return false;
    }
    return true;
}

// Right-click menu for one param: exact value, slider range, and (when a
// timeline is wired in) keying. Assumes the caller has pushed the param's ID.
static void render_param_popup(ShaderParam& p, Timeline* tl, const char* shader_file,
                               bool animatable, bool animated) {
    constexpr float W = 190.0f;

    ImGui::TextDisabled("%s", p.name.c_str());
    ImGui::Separator();

    // While this popup is open the timeline must stop driving the param, or
    // apply() overwrites every edit made here at the top of the next frame.
    // The slider's own IsItemActive() latch can't cover us — these are
    // different items, and a button click writes across a frame boundary.
    if (animated) {
        tl->editing_shader = shader_file;
        tl->editing_param = p.name;
    }

    // Typed value, deliberately unclamped: overshooting the range is how you
    // discover you want a wider one, and "Fit to value" turns that into one.
    if (p.type != ShaderParam::Enum) {
        ImGui::SetNextItemWidth(W);
        if (p.is_color) {
            // NoOptions: ImGui's own right-click menu on the swatch would
            // compete with this popup for the same click.
            if (p.component_count == 4) {
                ImGui::ColorEdit4("value", p.current_val, ImGuiColorEditFlags_NoOptions);
            } else {
                ImGui::ColorEdit3("value", p.current_val, ImGuiColorEditFlags_NoOptions);
            }
        } else if (p.type == ShaderParam::Int) {
            int v = (int)p.current_val[0];
            if (ImGui::InputInt("value", &v)) p.current_val[0] = (float)v;
        } else {
            ImGui::InputScalarN("value", ImGuiDataType_Float, p.current_val,
                                p.component_count, nullptr, nullptr, "%.6g");
        }
    }

    if (param_range_editable(p)) {
        bool is_int = (p.type == ShaderParam::Int);
        float fw = (W - 8.0f) * 0.5f;
        bool changed = false;

        ImGui::SetNextItemWidth(fw);
        if (is_int) {
            int lo = (int)p.min_val[0];
            if (ImGui::InputInt("##min", &lo, 0, 0)) { p.min_val[0] = (float)lo; changed = true; }
        } else {
            changed |= ImGui::InputFloat("##min", &p.min_val[0], 0, 0, "%.6g");
        }
        ImGui::SameLine(0, 8);
        ImGui::SetNextItemWidth(fw);
        if (is_int) {
            int hi = (int)p.max_val[0];
            if (ImGui::InputInt("##max", &hi, 0, 0)) { p.max_val[0] = (float)hi; changed = true; }
        } else {
            changed |= ImGui::InputFloat("##max", &p.max_val[0], 0, 0, "%.6g");
        }
        ImGui::SameLine(0, 8);
        ImGui::TextUnformatted("range");

        if (changed) {
            // An inverted range is harmless (ImGui draws a reversed slider),
            // but a non-finite one poisons the JSON — nlohmann writes NaN as
            // null, which the loader can't read back.
            if (!std::isfinite(p.min_val[0])) p.min_val[0] = p.decl_min[0];
            if (!std::isfinite(p.max_val[0])) p.max_val[0] = p.decl_max[0];
            // The widgets only ever use component 0; mirror so what we
            // serialize matches what's drawn.
            for (int c = 1; c < 4; c++) {
                p.min_val[c] = p.min_val[0];
                p.max_val[c] = p.max_val[0];
            }
        }

        ImGui::Checkbox("logarithmic", &p.logarithmic);

        float lo = std::min(p.min_val[0], p.max_val[0]);
        float hi = std::max(p.min_val[0], p.max_val[0]);
        bool out_of_range = false;
        for (int c = 0; c < p.component_count; c++) {
            if (p.current_val[c] < lo || p.current_val[c] > hi) out_of_range = true;
        }
        if (out_of_range) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.75f, 0.3f, 1.0f));
            ImGui::TextUnformatted("value is outside the range");
            ImGui::PopStyleColor();
        }

        if (ImGui::Button("Reset range")) {
            std::memcpy(p.min_val, p.decl_min, sizeof(p.min_val));
            std::memcpy(p.max_val, p.decl_max, sizeof(p.max_val));
            p.logarithmic = false;
        }
        ImGui::SameLine();
        if (ImGui::Button("Fit to value")) {
            for (int c = 0; c < p.component_count; c++) {
                lo = std::min(lo, p.current_val[c]);
                hi = std::max(hi, p.current_val[c]);
            }
            for (int c = 0; c < 4; c++) { p.min_val[c] = lo; p.max_val[c] = hi; }
        }
    }

    if (ImGui::Button("Reset value")) {
        std::memcpy(p.current_val, p.default_val, sizeof(p.current_val));
    }

    if (animatable) {
        ImGui::Separator();
        if (ImGui::MenuItem("Key at playhead")) tl->key_param(tl->playhead, shader_file, p);
        if (animated && ImGui::MenuItem("Remove animation")) {
            tl->remove_param(shader_file, p.name);
        }
    }
}

// One param's row: the widget, the animated tint, the timeline latch and
// auto-key hooks, and the right-click popup. Visibility is the caller's call.
static void render_param_row(ShaderParam& p, Timeline* tl, const char* shader_file,
                             bool animatable) {
    // Scope every widget to the param so the context menu below gets a
    // unique ID. Without this the multi-component sliders all collide:
    // SliderScalarN wraps its widgets in a group, and EndGroup() calls
    // ItemAdd(bb, 0), so LastItemData.ID is 0 for every one of them —
    // BeginPopupContextItem() with no str_id would key all their menus to
    // popup id 0 and render them stacked into one window.
    ImGui::PushID(p.name.c_str());

    bool animated = animatable && tl->has_track(shader_file, p.name);
    // Gold = the timeline drives this value; purple = it has tracks but
    // they're all muted, so the slider owns it (matches the dope sheet).
    bool muted = animated && !tl->track_active(shader_file, p.name);
    const ImVec4 anim_tint = muted ? ImVec4(0.72f, 0.48f, 0.95f, 1.0f)
                                   : ImVec4(0.95f, 0.85f, 0.35f, 1.0f);
    const ImU32 anim_overlay = muted ? IM_COL32(184, 122, 242, 255)
                                     : IM_COL32(242, 217, 89, 255);
    if (animated) ImGui::PushStyleColor(ImGuiCol_Text, anim_tint);

    const bool compact = g_compact_param_labels;

    // An overridden range is otherwise invisible until you right-click,
    // and it silently outranks the shader's declaration.
    char name_buf[96];
    std::snprintf(name_buf, sizeof(name_buf), "%s%s", p.name.c_str(),
                  has_range_override(p) ? " *" : "");

    // "###p" fixes the widget ID regardless of the visible text, so the
    // override marker can appear mid-drag without ImGui losing the item.
    // In compact mode the display part is empty and the name is painted
    // over the bar instead.
    char label[128];
    if (compact) std::snprintf(label, sizeof(label), "###p");
    else         std::snprintf(label, sizeof(label), "%s###p", name_buf);

    // Reclaim the space the label used to occupy. Colors are excluded:
    // NoInputs sizes the swatch to a fixed square, so their name goes
    // beside it (below) rather than on top of it.
    if (compact && !p.is_color) ImGui::SetNextItemWidth(-FLT_MIN);

    switch (p.type) {
        case ShaderParam::Float:
            ImGui::SliderFloat(label, &p.current_val[0],
                               p.min_val[0], p.max_val[0], slider_fmt(p), slider_flags(p));
            break;
        case ShaderParam::Int: {
            int v = (int)p.current_val[0];
            if (ImGui::SliderInt(label, &v,
                                 (int)p.min_val[0], (int)p.max_val[0], "%d", slider_flags(p))) {
                p.current_val[0] = (float)v;
            }
            break;
        }
        case ShaderParam::Enum: {
            int v = std::clamp((int)p.current_val[0], 0, (int)p.labels.size() - 1);
            // A combo's preview text is already left-aligned inside the
            // frame, so the name goes in the preview rather than being
            // painted on top of it — an overlay would collide.
            char preview[160];
            if (compact) {
                std::snprintf(preview, sizeof(preview), "%s:  %s",
                              p.name.c_str(), p.labels[v].c_str());
            } else {
                std::snprintf(preview, sizeof(preview), "%s", p.labels[v].c_str());
            }
            if (ImGui::BeginCombo(label, preview)) {
                for (int i = 0; i < (int)p.labels.size(); i++) {
                    if (ImGui::Selectable(p.labels[i].c_str(), i == v)) {
                        p.current_val[0] = (float)i;
                    }
                }
                ImGui::EndCombo();
            }
            break;
        }
        case ShaderParam::Float2:
            ImGui::SliderFloat2(label, p.current_val,
                                p.min_val[0], p.max_val[0], slider_fmt(p), slider_flags(p));
            break;
        case ShaderParam::Float3:
            if (p.is_color) {
                ImGui::ColorEdit3(label, p.current_val, color_flags(compact));
            } else {
                ImGui::SliderFloat3(label, p.current_val,
                                    p.min_val[0], p.max_val[0], slider_fmt(p), slider_flags(p));
            }
            break;
        case ShaderParam::Float4:
            if (p.is_color) {
                ImGui::ColorEdit4(label, p.current_val, color_flags(compact));
            } else {
                ImGui::SliderFloat4(label, p.current_val,
                                    p.min_val[0], p.max_val[0], slider_fmt(p), slider_flags(p));
            }
            break;
    }

    // Draw-list only, so it doesn't disturb LastItemData below. Enums
    // carry their name in the preview text and colors get theirs beside
    // the swatch, so neither wants an overlay.
    if (compact && !p.is_color && p.type != ShaderParam::Enum) {
        overlay_label(name_buf, animated ? anim_overlay : ImGui::GetColorU32(ImGuiCol_Text));
    }

    if (animated) ImGui::PopStyleColor();

    // These three all read g.LastItemData from the widget above, so
    // nothing may submit an item between it and them.
    if (animatable) {
        // Held slider: tell the timeline to stop driving this channel, or
        // the curve overwrites the drag at the top of the next frame.
        if (ImGui::IsItemActive()) {
            tl->editing_shader = shader_file;
            tl->editing_param = p.name;
        }
        if (tl->auto_key && ImGui::IsItemDeactivatedAfterEdit()) {
            tl->key_param(tl->playhead, shader_file, p);
        }
    }

    // Explicit str_id — see the PushID comment above.
    if (ImGui::BeginPopupContextItem("##ctx")) {
        render_param_popup(p, tl, shader_file, animatable, animated);
        ImGui::EndPopup();
    } else if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
        ImGui::SetTooltip(animatable ? "Right-click: value, range, animate"
                                     : "Right-click: value, range");
    }

    // A color's name sits next to its swatch. This submits an item, so it
    // has to come after everything above that reads LastItemData.
    if (compact && p.is_color) {
        ImGui::SameLine(0, ImGui::GetStyle().ItemInnerSpacing.x);
        if (animated) ImGui::PushStyleColor(ImGuiCol_Text, anim_tint);
        ImGui::TextUnformatted(name_buf);
        if (animated) ImGui::PopStyleColor();
    }

    ImGui::PopID();
}

void render_shader_params(std::vector<ShaderParam>& params, const ParamContext& ctx,
                          Timeline* tl, const char* shader_file) {
    const bool animatable = tl && shader_file;

    // Draw in `@group` sections: groups in first-appearance order, params in
    // declaration order within each. A display-only reordering — slot order,
    // which the shaders read by index, is untouched, so related params can
    // sit together despite the append-only slot rule.
    std::vector<std::string> groups;
    for (const auto& p : params) {
        if (std::find(groups.begin(), groups.end(), p.group) == groups.end()) {
            groups.push_back(p.group);
        }
    }

    for (const auto& g : groups) {
        bool header_drawn = false;
        for (auto& p : params) {
            if (p.group != g) continue;
            // Hidden params keep their value and still reach the GPU — the
            // shader just ignores them for the current fractal, so nothing
            // resets.
            if (!param_visible(p, params, ctx)) continue;
            // Deferred to the first visible param so a section that is
            // entirely hidden vanishes whole, header included.
            if (!header_drawn && !g.empty()) {
                ImGui::SeparatorText(g.c_str());
                header_drawn = true;
            }
            render_param_row(p, tl, shader_file, animatable);
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
