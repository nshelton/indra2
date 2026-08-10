#include "timeline.h"
#include "shader_manager.h"
#include "imgui.h"
#include <algorithm>
#include <cmath>
#include <cstdio>

// ---- Dope sheet + curve editor ----
//
// Drawn with ImDrawList primitives, but every interactive element is a real
// ImGui::InvisibleButton so hover/active/drag and context popups come from
// ImGui rather than hand-rolled hit testing. The background scrub bar is
// submitted first with SetNextItemAllowOverlap so the key diamonds on top of
// it still receive the mouse.

namespace {

constexpr float LABEL_W  = 180.0f;   // left column
constexpr float RULER_H  = 24.0f;
constexpr float ROW_H    = 20.0f;
constexpr float KEY_R    = 5.5f;     // diamond half-diagonal
constexpr float CURVE_H  = 180.0f;

const ImU32 COL_BG        = IM_COL32(28, 28, 32, 255);
const ImU32 COL_ROW_ALT   = IM_COL32(38, 38, 44, 255);
const ImU32 COL_GRID      = IM_COL32(60, 60, 68, 255);
const ImU32 COL_GRID_SEC  = IM_COL32(90, 90, 100, 255);
const ImU32 COL_PLAYHEAD  = IM_COL32(255, 90, 70, 255);
const ImU32 COL_KEY       = IM_COL32(220, 200, 90, 255);
const ImU32 COL_KEY_SEL   = IM_COL32(255, 255, 255, 255);
const ImU32 COL_KEY_STEP  = IM_COL32(120, 190, 230, 255);
const ImU32 COL_KEY_EDGE  = IM_COL32(20, 20, 24, 255);
const ImU32 COL_CAM       = IM_COL32(120, 220, 150, 255);
const ImU32 COL_TEXT_DIM  = IM_COL32(150, 150, 158, 255);
const ImU32 COL_MUTED     = IM_COL32(178, 112, 240, 255);  // parked track: slider owns the value

struct Row {
    enum Kind { Camera, Group, Component };
    Kind kind;
    std::string shader;
    std::string param;
    std::string label;
    std::string group_key;
    int component = -1;
    int track_index = -1;          // Component rows only
    const ShaderParam* sp = nullptr;
};

void draw_diamond(ImDrawList* dl, ImVec2 c, float r, ImU32 fill, ImU32 edge) {
    ImVec2 p[4] = {{c.x, c.y - r}, {c.x + r, c.y}, {c.x, c.y + r}, {c.x - r, c.y}};
    dl->AddQuadFilled(p[0], p[1], p[2], p[3], fill);
    dl->AddQuad(p[0], p[1], p[2], p[3], edge, 1.0f);
}

// Tick spacing that lands on 1/2/5 x 10^n and keeps ticks ~90px apart.
float nice_step(float span, float px, float target_px) {
    if (px < 1.0f) return span;
    float raw = span * (target_px / px);
    float mag = std::pow(10.0f, std::floor(std::log10(std::max(raw, 1e-6f))));
    float n = raw / mag;
    float m = n < 1.5f ? 1.0f : (n < 3.5f ? 2.0f : (n < 7.5f ? 5.0f : 10.0f));
    return m * mag;
}

bool interp_menu(Interp& mode) {
    bool changed = false;
    static const Interp modes[] = {Interp::Step, Interp::Linear, Interp::Smooth, Interp::Bezier};
    for (Interp m : modes) {
        if (ImGui::MenuItem(interp_name(m), nullptr, mode == m)) {
            mode = m;
            changed = true;
        }
    }
    return changed;
}

// ---- Multi-key selection ----
// Selection is the `selected` flag on Key/CameraKey, so it survives the
// sorts and inserts that would orphan stored indices.

void clear_selection(Timeline& tl) {
    for (auto& tr : tl.tracks)
        for (auto& k : tr.keys) k.selected = false;
    for (auto& k : tl.cam_keys) k.selected = false;
}

bool any_selected(const Timeline& tl) {
    for (const auto& tr : tl.tracks)
        for (const auto& k : tr.keys)
            if (k.selected) return true;
    for (const auto& k : tl.cam_keys)
        if (k.selected) return true;
    return false;
}

void delete_selected(Timeline& tl) {
    for (auto& tr : tl.tracks)
        std::erase_if(tr.keys, [](const Key& k) { return k.selected; });
    std::erase_if(tl.cam_keys, [](const CameraKey& k) { return k.selected; });
}

void copy_selection(const Timeline& tl, TimelineUI& ui) {
    float t0 = 1e9f;
    for (const auto& tr : tl.tracks)
        for (const auto& k : tr.keys)
            if (k.selected) t0 = std::min(t0, k.t);
    for (const auto& k : tl.cam_keys)
        if (k.selected) t0 = std::min(t0, k.t);
    if (t0 >= 1e9f) return;  // nothing selected; keep the old clipboard

    ui.clip_keys.clear();
    ui.clip_cam.clear();
    for (const auto& tr : tl.tracks) {
        for (const auto& k : tr.keys) {
            if (!k.selected) continue;
            TimelineUI::ClipKey ck{tr.shader, tr.param, tr.component, k};
            ck.key.t -= t0;
            ck.key.selected = false;
            ui.clip_keys.push_back(std::move(ck));
        }
    }
    for (const auto& k : tl.cam_keys) {
        if (!k.selected) continue;
        CameraKey ck = k;
        ck.t -= t0;
        ck.selected = false;
        ui.clip_cam.push_back(ck);
    }
}

// Pastes at the playhead, earliest key first, relative offsets intact. The
// pasted keys become the selection so a follow-up drag can place them.
// Anything already inside a pasted key's merge window is replaced, mirroring
// Track::insert. Returns false when the clipboard is empty.
bool paste_clipboard(Timeline& tl, TimelineUI& ui) {
    if (ui.clip_keys.empty() && ui.clip_cam.empty()) return false;
    clear_selection(tl);
    float window = tl.frame_time() * 0.5f;

    for (const auto& ck : ui.clip_keys) {
        Track* tr = tl.find(ck.shader, ck.param, ck.component);
        if (!tr) {
            // Track was deleted since the copy — recreate it, muted-off
            // defaults and all, so the paste never silently drops keys.
            Track nt;
            nt.shader = ck.shader;
            nt.param = ck.param;
            nt.component = ck.component;
            tl.tracks.push_back(std::move(nt));
            tr = &tl.tracks.back();
        }
        Key k = ck.key;
        k.t = std::clamp(tl.playhead + k.t, 0.0f, tl.duration);
        k.selected = true;
        std::erase_if(tr->keys, [&](const Key& e) { return std::fabs(e.t - k.t) <= window; });
        auto it = std::upper_bound(tr->keys.begin(), tr->keys.end(), k.t,
                                   [](float x, const Key& e) { return x < e.t; });
        tr->keys.insert(it, k);
    }
    for (const auto& sk : ui.clip_cam) {
        CameraKey k = sk;
        k.t = std::clamp(tl.playhead + k.t, 0.0f, tl.duration);
        k.selected = true;
        std::erase_if(tl.cam_keys,
                      [&](const CameraKey& e) { return std::fabs(e.t - k.t) <= window; });
        auto it = std::upper_bound(tl.cam_keys.begin(), tl.cam_keys.end(), k.t,
                                   [](float x, const CameraKey& e) { return x < e.t; });
        tl.cam_keys.insert(it, k);
    }
    return true;
}

// Dragging doesn't keep keys sorted (flags make that safe, and eval degrades
// gracefully on a transiently unsorted track); restore the invariant when
// the drag ends.
void sort_all_keys(Timeline& tl) {
    for (auto& tr : tl.tracks)
        std::sort(tr.keys.begin(), tr.keys.end(),
                  [](const Key& a, const Key& b) { return a.t < b.t; });
    std::sort(tl.cam_keys.begin(), tl.cam_keys.end(),
              [](const CameraKey& a, const CameraKey& b) { return a.t < b.t; });
}

// ---- Camera curve editor ----
// All eight camera channels as curves. Each channel normalizes to its own
// key range: target coordinates, log-distance and fov live on wildly
// different scales, and a shared axis would flatten most of them to lines.

struct CamChannel { const char* name; ImU32 col; };
constexpr CamChannel CAM_CHANNELS[8] = {
    {"tgt.x", IM_COL32(235, 96, 96, 255)},
    {"tgt.y", IM_COL32(110, 220, 110, 255)},
    {"tgt.z", IM_COL32(120, 160, 250, 255)},
    {"dir.x", IM_COL32(205, 140, 140, 255)},
    {"dir.y", IM_COL32(150, 200, 150, 255)},
    {"dir.z", IM_COL32(155, 175, 225, 255)},
    {"log_d", IM_COL32(230, 200, 90, 255)},
    {"fov",   IM_COL32(210, 130, 235, 255)},
};

float cam_channel_get(const CameraKey& k, int c) {
    switch (c) {
        case 0: case 1: case 2: return k.target[c];
        case 3: case 4: case 5: return k.dir[c - 3];
        case 6: return k.log_dist;
        default: return k.fov;
    }
}

void cam_channel_set(CameraKey& k, int c, float v) {
    if (c < 3) {
        k.target[c] = v;
    } else if (c < 6) {
        // dir lives on the unit sphere: write the component, renormalize.
        // The drag fights the normalization a little, but the stored key is
        // always valid. A degenerate edit keeps the previous axis.
        k.dir[c - 3] = v;
        float len = std::sqrt(k.dir[0] * k.dir[0] + k.dir[1] * k.dir[1] + k.dir[2] * k.dir[2]);
        if (len > 1e-6f) { k.dir[0] /= len; k.dir[1] /= len; k.dir[2] /= len; }
    } else if (c == 6) {
        k.log_dist = v;
    } else {
        k.fov = std::clamp(v, 0.05f, 3.0f);
    }
}

// Legend + toggles down the label column, curves and draggable keys to the
// right. Horizontal key drag moves the key in time (all channels — it is one
// key); vertical edits the grabbed channel. Returns true on any mutation.
bool draw_camera_curves(Timeline& tl, TimelineUI& ui, Camera& cam, ImDrawList* dl,
                        ImVec2 cp, float cw, float view_start, float view_end, float ft) {
    float span = std::max(view_end - view_start, 1e-6f);
    float cx0 = cp.x + LABEL_W;
    float cvw = cw - LABEL_W;
    auto t_to_x = [&](float t) { return cx0 + (t - view_start) / span * cvw; };
    auto x_to_t = [&](float x) { return view_start + (x - cx0) / cvw * span; };
    auto snap = [&](float t) {
        if (ImGui::GetIO().KeyShift) return t;
        return std::round(t / ft) * ft;
    };

    // Per-channel value range over the keys, padded; flat channels get ±0.5.
    float cmin[8], cmax[8];
    for (int c = 0; c < 8; c++) { cmin[c] = 1e9f; cmax[c] = -1e9f; }
    for (const auto& k : tl.cam_keys) {
        for (int c = 0; c < 8; c++) {
            float v = cam_channel_get(k, c);
            cmin[c] = std::min(cmin[c], v);
            cmax[c] = std::max(cmax[c], v);
        }
    }
    for (int c = 0; c < 8; c++) {
        if (cmax[c] - cmin[c] < 1e-6f) { cmin[c] -= 0.5f; cmax[c] += 0.5f; }
        float pad = (cmax[c] - cmin[c]) * 0.08f;
        cmin[c] -= pad;
        cmax[c] += pad;
    }
    auto v_to_y = [&](int c, float v) {
        return cp.y + CURVE_H - (v - cmin[c]) / (cmax[c] - cmin[c]) * CURVE_H;
    };
    auto y_to_v = [&](int c, float y) {
        return cmin[c] + (cp.y + CURVE_H - y) / CURVE_H * (cmax[c] - cmin[c]);
    };

    // Channel legend/toggles in the label column.
    for (int c = 0; c < 8; c++) {
        ImGui::SetCursorScreenPos(ImVec2(cp.x + 6, cp.y + 4 + c * 21.0f));
        bool on = (ui.cam_mask >> c) & 1u;
        ImGui::PushID(50000 + c);
        ImGui::PushStyleColor(ImGuiCol_Text, CAM_CHANNELS[c].col);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 0));
        if (ImGui::Checkbox(CAM_CHANNELS[c].name, &on)) {
            ui.cam_mask = on ? (ui.cam_mask | (1u << c)) : (ui.cam_mask & ~(1u << c));
        }
        ImGui::PopStyleVar();
        ImGui::PopStyleColor();
        ImGui::PopID();
    }

    // Curves through eval_camera_channels — the code playback runs, so slerp
    // kinks and step holds appear exactly as they will render.
    int steps = (int)std::min(cvw, 384.0f);
    std::vector<ImVec2> pts;
    CameraKey s;
    for (int c = 0; c < 8; c++) {
        if (!((ui.cam_mask >> c) & 1u)) continue;
        pts.clear();
        pts.reserve(steps + 1);
        for (int i = 0; i <= steps; i++) {
            float t = view_start + span * ((float)i / (float)steps);
            tl.eval_camera_channels(t, s);
            pts.push_back(ImVec2(t_to_x(t), v_to_y(c, cam_channel_get(s, c))));
        }
        dl->AddPolyline(pts.data(), (int)pts.size(), CAM_CHANNELS[c].col, 0, 1.4f);
    }

    bool mutated = false;
    for (int c = 0; c < 8; c++) {
        if (!((ui.cam_mask >> c) & 1u)) continue;
        for (size_t k = 0; k < tl.cam_keys.size(); k++) {
            CameraKey& ck = tl.cam_keys[k];
            float x = t_to_x(ck.t);
            float y = v_to_y(c, cam_channel_get(ck, c));
            if (x < cx0 - 8 || x > cx0 + cvw + 8) continue;
            ImGui::PushID(60000 + c * 4096 + (int)k);
            ImGui::SetCursorScreenPos(ImVec2(x - KEY_R, y - KEY_R));
            ImGui::InvisibleButton("##cck", ImVec2(KEY_R * 2, KEY_R * 2));
            if (ImGui::IsItemActivated()) ui.sel_key = (int)k;
            if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                ck.t = std::clamp(snap(x_to_t(ImGui::GetIO().MousePos.x)), 0.0f, tl.duration);
                cam_channel_set(ck, c, y_to_v(c, ImGui::GetIO().MousePos.y));
                mutated = true;
            }
            if (ImGui::IsItemDeactivatedAfterEdit()) {
                std::sort(tl.cam_keys.begin(), tl.cam_keys.end(),
                          [](const CameraKey& a, const CameraKey& b) { return a.t < b.t; });
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("%s @ %.3fs = %.4f", CAM_CHANNELS[c].name, ck.t,
                                  cam_channel_get(ck, c));
            }
            bool del = false;
            if (ImGui::BeginPopupContextItem("##cckm")) {
                ImGui::TextDisabled("camera key %.3fs", ck.t);
                ImGui::Separator();
                if (interp_menu(ck.interp)) mutated = true;
                ImGui::Separator();
                if (ImGui::MenuItem("Go to")) tl.playhead = ck.t;
                if (ImGui::MenuItem("Replace with current pose")) {
                    tl.key_camera(ck.t, cam);
                    mutated = true;
                }
                if (ImGui::MenuItem("Delete")) del = true;
                ImGui::EndPopup();
            }
            bool sel = ck.selected || (ui.sel_camera && ui.sel_key == (int)k);
            draw_diamond(dl, ImVec2(x, y), KEY_R - 1.0f,
                         sel ? COL_KEY_SEL : CAM_CHANNELS[c].col, COL_KEY_EDGE);
            ImGui::PopID();
            if (del) {
                tl.remove_camera_key((int)k);
                ui.sel_key = -1;
                // Key indices are stale for every diamond after this one —
                // bail and let the next frame rebuild.
                return true;
            }
        }
    }
    return mutated;
}

// Every ShaderParam the timeline can see, in shader declaration order.
void collect_rows(Timeline& tl, TimelineUI& ui, ShaderManager& shaders, std::vector<Row>& rows) {
    rows.push_back({Row::Camera, "", "", "Camera", "", -1, -1, nullptr});

    for (const auto& entry : shaders.entries()) {
        for (const auto& p : entry.params) {
            if (!tl.has_track(entry.filename, p.name)) continue;
            std::string gk = entry.filename + "|" + p.name;
            Row g{Row::Group, entry.filename, p.name, p.name, gk, -1, -1, &p};
            rows.push_back(g);
            if (p.component_count > 1 && ui.is_expanded(gk)) {
                static const char* comp_names[4] = {"x", "y", "z", "w"};
                for (int c = 0; c < p.component_count; c++) {
                    int ti = -1;
                    for (size_t i = 0; i < tl.tracks.size(); i++) {
                        const Track& t = tl.tracks[i];
                        if (t.shader == entry.filename && t.param == p.name && t.component == c) {
                            ti = (int)i;
                            break;
                        }
                    }
                    if (ti < 0) continue;
                    rows.push_back({Row::Component, entry.filename, p.name,
                                    std::string("  ") + comp_names[c], gk, c, ti, &p});
                }
            }
        }
    }
}

}  // namespace

bool TimelineUI::is_expanded(const std::string& key) const {
    return std::find(expanded.begin(), expanded.end(), key) != expanded.end();
}

void TimelineUI::toggle_expanded(const std::string& key) {
    auto it = std::find(expanded.begin(), expanded.end(), key);
    if (it == expanded.end()) expanded.push_back(key);
    else expanded.erase(it);
}

void draw_timeline(Timeline& tl, TimelineUI& ui, ShaderManager& shaders, Camera& cam) {
    const float ft = tl.frame_time();

    // ---- Transport ----
    if (ImGui::Button(tl.playing ? "Pause" : "Play", ImVec2(58, 0))) tl.playing = !tl.playing;
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Space");
    ImGui::SameLine();
    if (ImGui::Button("Stop", ImVec2(48, 0))) {
        tl.playing = false;
        tl.playhead = 0.0f;
    }
    ImGui::SameLine();
    ImGui::Checkbox("Loop", &tl.loop);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(70);
    // AlwaysClamp: ctrl-click typing bypasses drag bounds otherwise, and the
    // render job divides by fps — 0 would pin every frame's time to NaN.
    if (ImGui::DragFloat("fps", &tl.fps, 0.5f, 1.0f, 240.0f, "%.0f",
                         ImGuiSliderFlags_AlwaysClamp)) tl.revision++;
    ImGui::SameLine();
    ImGui::SetNextItemWidth(70);
    if (ImGui::DragFloat("dur", &tl.duration, 0.1f, 0.1f, 3600.0f, "%.2fs",
                         ImGuiSliderFlags_AlwaysClamp)) tl.revision++;

    ImGui::SameLine();
    ImGui::Checkbox("Auto-key", &tl.auto_key);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Moving the camera or a slider writes a key at the playhead");
    }
    ImGui::SameLine();
    if (ImGui::Button("Key Camera")) tl.key_camera(tl.playhead, cam);
    ImGui::SameLine();
    ImGui::Checkbox("Drive cam", &tl.drive_camera);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Off: camera keys are ignored and you keep manual control");
    }

    ImGui::Text("t = %.3fs   frame %d / %d", tl.playhead, (int)std::lround(tl.playhead * tl.fps),
                (int)std::lround(tl.duration * tl.fps));
    ImGui::SameLine();
    ImGui::Checkbox("Curves", &ui.show_curves);

    // Keyboard edits on the selection: Delete/Backspace removes it,
    // Cmd/Ctrl+C/X/V copy, cut and paste (paste lands at the playhead).
    // Before collect_rows so the row list never sees deleted keys; gated on
    // text input so typing a preset name can't fire them.
    if (!ImGui::GetIO().WantTextInput &&
        ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows)) {
        bool have_sel = any_selected(tl);
        if (have_sel && (ImGui::IsKeyPressed(ImGuiKey_Delete, false) ||
                         ImGui::IsKeyPressed(ImGuiKey_Backspace, false))) {
            delete_selected(tl);
            ui.sel_track = -1;
            ui.sel_key = -1;
            tl.revision++;
        }
        if (have_sel && ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_C)) {
            copy_selection(tl, ui);
        }
        if (have_sel && ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_X)) {
            copy_selection(tl, ui);
            delete_selected(tl);
            ui.sel_track = -1;
            ui.sel_key = -1;
            tl.revision++;
        }
        if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_V)) {
            if (paste_clipboard(tl, ui)) tl.revision++;
        }
    }

    // ---- Rows ----
    std::vector<Row> rows;
    collect_rows(tl, ui, shaders, rows);

    // ---- Canvas geometry ----
    // The ruler is the animation: a full-span view follows duration edits,
    // and the view never extends past [0, duration] — keys can't live out
    // there, so there is nothing to show.
    const float full = std::max(tl.duration, 0.1f);
    bool was_full = ui.view_start <= 1e-4f && ui.view_end == ui.fitted_duration;
    if (ui.fitted_duration < 0.0f || was_full) {
        ui.view_start = 0.0f;
        ui.view_end   = full;
    }
    ui.fitted_duration = full;
    ui.view_start = std::clamp(ui.view_start, 0.0f, full);
    ui.view_end   = std::clamp(ui.view_end,   0.0f, full);
    if (ui.view_end <= ui.view_start + 1e-4f) {
        ui.view_start = 0.0f;
        ui.view_end   = full;
    }
    ImVec2 canvas_p = ImGui::GetCursorScreenPos();
    float canvas_w = std::max(ImGui::GetContentRegionAvail().x, LABEL_W + 60.0f);
    float track_x0 = canvas_p.x + LABEL_W;
    float track_w  = canvas_w - LABEL_W;

    const float span = ui.view_end - ui.view_start;
    auto t_to_x = [&](float t) { return track_x0 + (t - ui.view_start) / span * track_w; };
    auto x_to_t = [&](float x) { return ui.view_start + (x - track_x0) / track_w * span; };
    auto snap = [&](float t) {
        if (ImGui::GetIO().KeyShift) return t;
        return std::round(t / ft) * ft;
    };

    // Dragging any selected key moves the whole selection rigidly. Snapping
    // works on the accumulated mouse motion (Shift = free), and the step is
    // clamped so the selection's extremes stay inside [0, duration] without
    // squashing the offsets between keys.
    auto drag_selected = [&]() -> bool {
        ui.drag_accum += ImGui::GetIO().MouseDelta.x * (span / track_w);
        float target = ImGui::GetIO().KeyShift ? ui.drag_accum
                                               : std::round(ui.drag_accum / ft) * ft;
        float step = target - ui.drag_applied;
        if (step == 0.0f) return false;
        float mn = 1e9f, mx = -1e9f;
        for (const auto& tr : tl.tracks)
            for (const auto& k : tr.keys)
                if (k.selected) { mn = std::min(mn, k.t); mx = std::max(mx, k.t); }
        for (const auto& k : tl.cam_keys)
            if (k.selected) { mn = std::min(mn, k.t); mx = std::max(mx, k.t); }
        if (mx < mn) return false;
        step = std::clamp(step, -mn, tl.duration - mx);
        if (step == 0.0f) return false;
        for (auto& tr : tl.tracks)
            for (auto& k : tr.keys)
                if (k.selected) k.t += step;
        for (auto& k : tl.cam_keys)
            if (k.selected) k.t += step;
        ui.drag_applied += step;
        return true;
    };

    ImDrawList* dl = ImGui::GetWindowDrawList();

    // ---- Ruler ----
    ImVec2 ruler_min = canvas_p;
    ImVec2 ruler_max = ImVec2(canvas_p.x + canvas_w, canvas_p.y + RULER_H);
    dl->AddRectFilled(ruler_min, ruler_max, COL_BG);

    ImGui::SetCursorScreenPos(ImVec2(track_x0, canvas_p.y));
    ImGui::InvisibleButton("##ruler", ImVec2(track_w, RULER_H));
    bool ruler_active = ImGui::IsItemActive();
    bool ruler_hovered = ImGui::IsItemHovered();
    if (ruler_active) {
        tl.playhead = std::clamp(snap(x_to_t(ImGui::GetIO().MousePos.x)), 0.0f, tl.duration);
        tl.playing = false;
    }

    // Zoom on wheel over the ruler, anchored at the cursor; middle-drag pans.
    // Both clamp to [0, duration]: fully zoomed out IS the animation, which
    // also re-glues the view to duration edits (see canvas geometry above).
    if (ruler_hovered) {
        float wheel = ImGui::GetIO().MouseWheel;
        if (wheel != 0.0f) {
            float anchor = x_to_t(ImGui::GetIO().MousePos.x);
            float k = std::pow(0.85f, wheel);
            ui.view_start = std::max(anchor + (ui.view_start - anchor) * k, 0.0f);
            ui.view_end   = std::min(anchor + (ui.view_end   - anchor) * k, full);
        }
    }
    if (ImGui::IsMouseDragging(ImGuiMouseButton_Middle) && ruler_hovered) {
        float dt = ImGui::GetIO().MouseDelta.x / track_w * span;
        ui.view_start -= dt;
        ui.view_end   -= dt;
        // Slide the window back inside the animation without changing its width.
        if (ui.view_start < 0.0f) {
            ui.view_end -= ui.view_start;
            ui.view_start = 0.0f;
        }
        if (ui.view_end > full) {
            ui.view_start = std::max(ui.view_start - (ui.view_end - full), 0.0f);
            ui.view_end   = full;
        }
    }

    {
        float step = nice_step(span, track_w, 90.0f);
        float t0 = std::floor(ui.view_start / step) * step;
        char buf[32];
        for (float t = t0; t <= ui.view_end + step * 0.5f; t += step) {
            float x = t_to_x(t);
            if (x < track_x0 - 1 || x > track_x0 + track_w + 1) continue;
            dl->AddLine(ImVec2(x, ruler_min.y + RULER_H * 0.55f), ImVec2(x, ruler_max.y), COL_GRID_SEC);
            std::snprintf(buf, sizeof(buf), step >= 1.0f ? "%.0fs" : "%.2fs", t);
            dl->AddText(ImVec2(x + 3, ruler_min.y + 3), COL_TEXT_DIM, buf);
        }
    }
    dl->AddLine(ImVec2(canvas_p.x, ruler_max.y - 1), ImVec2(ruler_max.x, ruler_max.y - 1), COL_GRID);

    // ---- Track region (scrolls) ----
    ImGui::SetCursorScreenPos(ImVec2(canvas_p.x, ruler_max.y));
    float rows_h = std::max(ROW_H * rows.size() + 4.0f, ROW_H * 3.0f);
    float avail_h = ImGui::GetContentRegionAvail().y - (ui.show_curves ? CURVE_H + 8.0f : 0.0f);
    float child_h = std::clamp(rows_h, ROW_H * 3.0f, std::max(avail_h, ROW_H * 3.0f));

    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImGui::ColorConvertU32ToFloat4(COL_BG));
    ImGui::BeginChild("##tracks", ImVec2(canvas_w, child_h), ImGuiChildFlags_None,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    ImDrawList* tdl = ImGui::GetWindowDrawList();
    ImVec2 tracks_p = ImGui::GetCursorScreenPos();

    // Background surface, submitted first so key diamonds overlap it. Drag
    // draws a marquee that selects the keys inside it (Shift adds to the
    // selection); a plain click clears the selection and places the
    // playhead. Continuous scrub-dragging lives on the ruler.
    ImGui::SetNextItemAllowOverlap();
    ImGui::SetCursorScreenPos(ImVec2(track_x0, tracks_p.y));
    ImGui::InvisibleButton("##scrub", ImVec2(track_w, child_h));
    {
        ImGuiIO& io = ImGui::GetIO();
        if (ImGui::IsItemActivated()) {
            ui.marquee = false;
            ui.marquee_x0 = io.MousePos.x;
            ui.marquee_y0 = io.MousePos.y;
        }
        if (ImGui::IsItemActive() && !ui.marquee &&
            ImGui::IsMouseDragging(ImGuiMouseButton_Left, 4.0f)) {
            ui.marquee = true;
        }
        if (ImGui::IsItemDeactivated()) {
            if (ui.marquee) {
                ui.marquee = false;
            } else {
                clear_selection(tl);
                tl.playhead = std::clamp(snap(x_to_t(io.MousePos.x)), 0.0f, tl.duration);
                tl.playing = false;
            }
        }
    }
    bool marquee_live = ui.marquee && ImGui::IsItemActive();
    float mq_x0 = std::min(ui.marquee_x0, ImGui::GetIO().MousePos.x);
    float mq_x1 = std::max(ui.marquee_x0, ImGui::GetIO().MousePos.x);
    float mq_y0 = std::min(ui.marquee_y0, ImGui::GetIO().MousePos.y);
    float mq_y1 = std::max(ui.marquee_y0, ImGui::GetIO().MousePos.y);
    // Replace-mode marquee rebuilds the selection every frame from whatever
    // is inside the rectangle; Shift keeps prior flags (additive, sticky).
    if (marquee_live && !ImGui::GetIO().KeyShift) clear_selection(tl);

    // Vertical time grid behind the rows.
    {
        float step = nice_step(span, track_w, 90.0f);
        for (float t = std::floor(ui.view_start / step) * step; t <= ui.view_end; t += step) {
            float x = t_to_x(t);
            if (x < track_x0 || x > track_x0 + track_w) continue;
            tdl->AddLine(ImVec2(x, tracks_p.y), ImVec2(x, tracks_p.y + child_h), COL_GRID);
        }
    }

    bool mutated = false;
    // Deleting a key or a track invalidates the row list and the track indices
    // captured in it. Rather than patch them up mid-frame, bail out of the row
    // loop and rebuild next frame.
    bool structural_change = false;

    for (size_t r = 0; r < rows.size(); r++) {
        const Row& row = rows[r];
        float ry = tracks_p.y + ROW_H * r;
        if (ry > tracks_p.y + child_h) break;

        if (r & 1) {
            tdl->AddRectFilled(ImVec2(canvas_p.x, ry), ImVec2(canvas_p.x + canvas_w, ry + ROW_H),
                               COL_ROW_ALT);
        }

        // --- Label column ---
        // Mute checkbox first: checked = the timeline drives this value,
        // unchecked = keys park (purple) and the slider — or the mouse, for
        // the camera — owns it. Ids key on group_key/component, stable across
        // row-index shifts; the group box toggles every component together.
        bool row_muted = false;
        ImGui::SetCursorScreenPos(ImVec2(canvas_p.x + 4, ry + 3));
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 0));
        if (row.kind == Row::Camera) {
            ImGui::PushID("cam_on");
            if (ImGui::Checkbox("##on", &tl.drive_camera)) mutated = true;
            ImGui::PopID();
            row_muted = !tl.drive_camera;
        } else if (row.kind == Row::Component) {
            Track& tr = tl.tracks[row.track_index];
            ImGui::PushID(row.group_key.c_str());
            ImGui::PushID(row.component);
            if (ImGui::Checkbox("##on", &tr.enabled)) mutated = true;
            ImGui::PopID();
            ImGui::PopID();
            row_muted = !tr.enabled;
        } else {
            bool any = false, all_on = true, any_on = false;
            for (const auto& t : tl.tracks) {
                if (t.shader != row.shader || t.param != row.param) continue;
                any = true;
                all_on &= t.enabled;
                any_on |= t.enabled;
            }
            row_muted = !any_on;  // purple only when every component is parked
            bool on = any && all_on;
            ImGui::PushID(row.group_key.c_str());
            if (ImGui::Checkbox("##on", &on)) {
                for (auto& t : tl.tracks) {
                    if (t.shader == row.shader && t.param == row.param) t.enabled = on;
                }
                mutated = true;
            }
            ImGui::PopID();
        }
        ImGui::PopStyleVar();
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(row_muted ? "Muted: the value is set by hand, keys are parked"
                                        : "Animating — uncheck to set the value by hand");
        }
        ImGui::SameLine(0, 4);

        if (row.kind == Row::Group && row.sp && row.sp->component_count > 1) {
            ImGui::PushID((int)r);
            if (ImGui::SmallButton(ui.is_expanded(row.group_key) ? "v" : ">")) {
                ui.toggle_expanded(row.group_key);
            }
            ImGui::PopID();
            ImGui::SameLine(0, 4);
        }
        ImU32 label_col = row_muted               ? COL_MUTED
                        : row.kind == Row::Camera ? COL_CAM
                                                  : IM_COL32(220, 220, 226, 255);
        ImGui::PushStyleColor(ImGuiCol_Text, label_col);
        ImGui::TextUnformatted(row.label.c_str());
        ImGui::PopStyleColor();

        // Popup id from group_key + component, not the label: labels collide
        // (every expanded float3 has a "  x" row, params can share a name
        // across shaders), and colliding ids merge those rows' menus so the
        // first row's "Delete track" acts from the other row's popup.
        char ctx_id[320];
        std::snprintf(ctx_id, sizeof(ctx_id), "%s#%d", row.group_key.c_str(), row.component);
        if (row.kind != Row::Camera && ImGui::BeginPopupContextItem(ctx_id)) {
            ImGui::TextDisabled("%s", row.shader.c_str());
            ImGui::Separator();
            if (ImGui::MenuItem("Key at playhead")) {
                for (auto& p : shaders.get_params_mut(row.shader)) {
                    if (p.name == row.param) { tl.key_param(tl.playhead, row.shader, p); break; }
                }
                mutated = true;
            }
            if (ImGui::MenuItem("Delete track")) {
                tl.remove_param(row.shader, row.param);
                ui.sel_track = -1;
                ui.sel_key = -1;
                mutated = true;
                structural_change = true;
            }
            ImGui::EndPopup();
        }
        if (structural_change) break;

        // --- Keys ---
        float cy = ry + ROW_H * 0.5f;

        if (row.kind == Row::Camera) {
            for (size_t k = 0; k < tl.cam_keys.size(); k++) {
                CameraKey& ck = tl.cam_keys[k];
                float x = t_to_x(ck.t);
                if (x < track_x0 - KEY_R || x > track_x0 + track_w + KEY_R) continue;
                if (marquee_live && x >= mq_x0 && x <= mq_x1 && cy >= mq_y0 && cy <= mq_y1) {
                    ck.selected = true;
                }
                ImGui::PushID((int)(r * 4096 + k));
                ImGui::SetCursorScreenPos(ImVec2(x - KEY_R, cy - KEY_R));
                ImGui::InvisibleButton("##ck", ImVec2(KEY_R * 2, KEY_R * 2));
                bool selected = ck.selected;
                if (ImGui::IsItemActivated()) {
                    ui.sel_camera = true;
                    ui.sel_key = (int)k;
                    ui.sel_track = -1;
                    // Grabbing an unselected key retargets the selection to
                    // it (Shift adds); grabbing a selected one keeps the set
                    // so the drag below moves the whole group.
                    if (!ck.selected) {
                        if (!ImGui::GetIO().KeyShift) clear_selection(tl);
                        ck.selected = true;
                        selected = true;
                    }
                    ui.drag_accum = ui.drag_applied = 0.0f;
                }
                if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                    if (drag_selected()) mutated = true;
                }
                if (ImGui::IsItemDeactivated()) sort_all_keys(tl);
                if (ImGui::BeginPopupContextItem("##ckm")) {
                    ImGui::TextDisabled("camera key %.3fs", tl.cam_keys[k].t);
                    ImGui::Separator();
                    if (interp_menu(tl.cam_keys[k].interp)) mutated = true;
                    ImGui::Separator();
                    if (ImGui::MenuItem("Go to")) tl.playhead = tl.cam_keys[k].t;
                    if (ImGui::MenuItem("Replace with current pose")) {
                        tl.key_camera(tl.cam_keys[k].t, cam);
                        mutated = true;
                    }
                    if (ImGui::MenuItem("Delete")) {
                        tl.remove_camera_key((int)k);
                        ui.sel_key = -1;
                        mutated = true;
                        structural_change = true;
                    }
                    ImGui::EndPopup();
                }
                if (structural_change) { ImGui::PopID(); break; }
                ImU32 fill = selected  ? COL_KEY_SEL
                           : row_muted ? COL_MUTED
                           : (tl.cam_keys[k].interp == Interp::Step ? COL_KEY_STEP : COL_CAM);
                draw_diamond(tdl, ImVec2(x, cy), KEY_R, fill, COL_KEY_EDGE);
                ImGui::PopID();
            }
            if (structural_change) break;
            continue;
        }

        // Group rows show the union of their components' key times and move
        // all of them together; component rows edit one channel.
        std::vector<float> times;
        std::vector<int> owner;   // track index per drawn key, -1 for group rows
        std::vector<int> key_idx;
        if (row.kind == Row::Component) {
            const Track& tr = tl.tracks[row.track_index];
            for (size_t k = 0; k < tr.keys.size(); k++) {
                times.push_back(tr.keys[k].t);
                owner.push_back(row.track_index);
                key_idx.push_back((int)k);
            }
        } else {
            for (const auto& tr : tl.tracks) {
                if (tr.shader != row.shader || tr.param != row.param) continue;
                for (const auto& k : tr.keys) {
                    bool dup = false;
                    for (float e : times) {
                        if (std::fabs(e - k.t) < ft * 0.25f) { dup = true; break; }
                    }
                    if (!dup) {
                        times.push_back(k.t);
                        owner.push_back(-1);
                        key_idx.push_back(-1);
                    }
                }
            }
            std::sort(times.begin(), times.end());
        }

        // The keys this row addresses at a given time: a group row spans
        // every component's key inside the merge window, a component row
        // just its own channel.
        auto row_keys_at = [&](float t, auto&& fn) {
            for (auto& tr : tl.tracks) {
                if (tr.shader != row.shader || tr.param != row.param) continue;
                if (row.kind == Row::Component && tr.component != row.component) continue;
                for (auto& key : tr.keys) {
                    if (std::fabs(key.t - t) < ft * 0.25f) fn(key);
                }
            }
        };

        for (size_t k = 0; k < times.size(); k++) {
            float x = t_to_x(times[k]);
            if (x < track_x0 - KEY_R || x > track_x0 + track_w + KEY_R) continue;
            if (marquee_live && x >= mq_x0 && x <= mq_x1 && cy >= mq_y0 && cy <= mq_y1) {
                row_keys_at(times[k], [](Key& key) { key.selected = true; });
            }
            ImGui::PushID((int)(r * 4096 + k));
            ImGui::SetCursorScreenPos(ImVec2(x - KEY_R, cy - KEY_R));
            ImGui::InvisibleButton("##k", ImVec2(KEY_R * 2, KEY_R * 2));

            bool selected = false;
            row_keys_at(times[k], [&](Key& key) { selected |= key.selected; });
            if (ImGui::IsItemActivated()) {
                ui.sel_camera = false;
                ui.sel_track = owner[k];
                ui.sel_key = key_idx[k];
                // Grabbing an unselected key retargets the selection to it
                // (Shift adds); a selected one keeps the set for the drag.
                if (!selected) {
                    if (!ImGui::GetIO().KeyShift) clear_selection(tl);
                    row_keys_at(times[k], [](Key& key) { key.selected = true; });
                    selected = true;
                }
                ui.drag_accum = ui.drag_applied = 0.0f;
                if (owner[k] < 0) ui.toggle_expanded(row.group_key);
            }
            if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                if (drag_selected()) mutated = true;
            }
            if (ImGui::IsItemDeactivated()) sort_all_keys(tl);

            if (ImGui::BeginPopupContextItem("##km")) {
                ImGui::TextDisabled("%s @ %.3fs", row.param.c_str(), times[k]);
                ImGui::Separator();
                Interp mode = Interp::Smooth;
                bool mode_found = false;
                for (const auto& tr : tl.tracks) {
                    if (tr.shader != row.shader || tr.param != row.param) continue;
                    if (row.kind == Row::Component && tr.component != row.component) continue;
                    for (const auto& key : tr.keys) {
                        if (std::fabs(key.t - times[k]) < ft * 0.25f) {
                            mode = key.interp;
                            mode_found = true;
                            break;
                        }
                    }
                    if (mode_found) break;
                }
                if (interp_menu(mode)) {
                    for (auto& tr : tl.tracks) {
                        if (tr.shader != row.shader || tr.param != row.param) continue;
                        if (row.kind == Row::Component && tr.component != row.component) continue;
                        for (auto& key : tr.keys) {
                            if (std::fabs(key.t - times[k]) < ft * 0.25f) key.interp = mode;
                        }
                    }
                    mutated = true;
                }
                ImGui::Separator();
                if (ImGui::MenuItem("Go to")) tl.playhead = times[k];
                if (ImGui::MenuItem("Delete")) {
                    for (auto& tr : tl.tracks) {
                        if (tr.shader != row.shader || tr.param != row.param) continue;
                        if (row.kind == Row::Component && tr.component != row.component) continue;
                        for (int ki = (int)tr.keys.size() - 1; ki >= 0; ki--) {
                            if (std::fabs(tr.keys[ki].t - times[k]) < ft * 0.25f) tr.erase_at(ki);
                        }
                    }
                    ui.sel_key = -1;
                    mutated = true;
                    structural_change = true;   // key_idx below is now stale
                }
                ImGui::EndPopup();
            }
            if (structural_change) { ImGui::PopID(); break; }

            // Dragging re-sorts the keys, so key_idx can point at a different
            // key than it did when the row was built. Bounds-check it: a wrong
            // colour for one frame is fine, an out-of-range read is not.
            Interp mode = Interp::Smooth;
            if (owner[k] >= 0 && key_idx[k] >= 0 &&
                key_idx[k] < (int)tl.tracks[owner[k]].keys.size()) {
                mode = tl.tracks[owner[k]].keys[key_idx[k]].interp;
            }
            ImU32 fill = selected  ? COL_KEY_SEL
                       : row_muted ? COL_MUTED
                       : (mode == Interp::Step ? COL_KEY_STEP : COL_KEY);
            draw_diamond(tdl, ImVec2(x, cy), KEY_R, fill, COL_KEY_EDGE);
            ImGui::PopID();
        }
        if (structural_change) break;
    }

    // Marquee rectangle, over the keys it is selecting.
    if (marquee_live) {
        tdl->AddRectFilled(ImVec2(mq_x0, mq_y0), ImVec2(mq_x1, mq_y1), IM_COL32(130, 170, 255, 32));
        tdl->AddRect(ImVec2(mq_x0, mq_y0), ImVec2(mq_x1, mq_y1), IM_COL32(130, 170, 255, 200));
    }

    // Playhead last so it draws over the keys.
    {
        float px = t_to_x(tl.playhead);
        if (px >= track_x0 && px <= track_x0 + track_w) {
            tdl->AddLine(ImVec2(px, tracks_p.y), ImVec2(px, tracks_p.y + child_h), COL_PLAYHEAD, 1.5f);
        }
    }
    ImGui::EndChild();
    ImGui::PopStyleColor();

    // Playhead over the ruler too.
    {
        float px = t_to_x(tl.playhead);
        if (px >= track_x0 && px <= track_x0 + track_w) {
            dl->AddLine(ImVec2(px, ruler_min.y), ImVec2(px, ruler_max.y), COL_PLAYHEAD, 1.5f);
            dl->AddTriangleFilled(ImVec2(px - 5, ruler_min.y), ImVec2(px + 5, ruler_min.y),
                                  ImVec2(px, ruler_min.y + 8), COL_PLAYHEAD);
        }
    }

    if (mutated) tl.revision++;

    // ---- Curve editor ----
    if (!ui.show_curves) return;

    ImGui::Spacing();
    ImVec2 cp = ImGui::GetCursorScreenPos();
    float cw = canvas_w;
    dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(cp, ImVec2(cp.x + cw, cp.y + CURVE_H), COL_BG);

    // Camera mode: selecting a camera key flips the curve view to the eight
    // camera channels. Selecting a param key flips it back.
    if (ui.sel_camera && !tl.cam_keys.empty()) {
        ImGui::SetCursorScreenPos(cp);
        ImGui::SetNextItemAllowOverlap();
        ImGui::InvisibleButton("##curvebg", ImVec2(cw, CURVE_H));
        dl->PushClipRect(cp, ImVec2(cp.x + cw, cp.y + CURVE_H), true);
        if (draw_camera_curves(tl, ui, cam, dl, cp, cw, ui.view_start, ui.view_end, ft)) {
            tl.revision++;
        }
        {
            float px = t_to_x(tl.playhead);
            if (px >= cp.x + LABEL_W && px <= cp.x + cw) {
                dl->AddLine(ImVec2(px, cp.y), ImVec2(px, cp.y + CURVE_H), COL_PLAYHEAD, 1.5f);
            }
        }
        dl->PopClipRect();
        dl->AddRect(cp, ImVec2(cp.x + cw, cp.y + CURVE_H), COL_GRID);
        ImGui::SetCursorScreenPos(ImVec2(cp.x, cp.y + CURVE_H));
        return;
    }

    // Which tracks to show: the selected key's param, else the first group.
    std::string cur_shader, cur_param;
    const ShaderParam* cur_sp = nullptr;
    if (!ui.sel_camera && ui.sel_track >= 0 && ui.sel_track < (int)tl.tracks.size()) {
        cur_shader = tl.tracks[ui.sel_track].shader;
        cur_param  = tl.tracks[ui.sel_track].param;
    } else {
        for (const auto& row : rows) {
            if (row.kind == Row::Group) { cur_shader = row.shader; cur_param = row.param; break; }
        }
    }
    for (const auto& row : rows) {
        if (row.kind == Row::Group && row.shader == cur_shader && row.param == cur_param) {
            cur_sp = row.sp;
            break;
        }
    }

    if (!cur_sp) {
        dl->AddText(ImVec2(cp.x + 8, cp.y + 8), COL_TEXT_DIM, "No animated parameter selected");
        ImGui::Dummy(ImVec2(cw, CURVE_H));
        return;
    }

    // Value axis from the param's declared range, padded so keys parked at a
    // limit aren't stuck on the border.
    float vmin = cur_sp->min_val[0], vmax = cur_sp->max_val[0];
    for (const auto& tr : tl.tracks) {
        if (tr.shader != cur_shader || tr.param != cur_param) continue;
        for (const auto& k : tr.keys) { vmin = std::min(vmin, k.v); vmax = std::max(vmax, k.v); }
    }
    if (vmax - vmin < 1e-6f) { vmin -= 0.5f; vmax += 0.5f; }
    float pad = (vmax - vmin) * 0.08f;
    vmin -= pad; vmax += pad;

    float cx0 = cp.x + LABEL_W;
    float cvw = cw - LABEL_W;
    auto ct_to_x = [&](float t) { return cx0 + (t - ui.view_start) / span * cvw; };
    auto cx_to_t = [&](float x) { return ui.view_start + (x - cx0) / cvw * span; };
    auto v_to_y = [&](float v) { return cp.y + CURVE_H - (v - vmin) / (vmax - vmin) * CURVE_H; };
    auto y_to_v = [&](float y) { return vmin + (cp.y + CURVE_H - y) / CURVE_H * (vmax - vmin); };

    dl->AddText(ImVec2(cp.x + 6, cp.y + 4), IM_COL32(220, 220, 226, 255), cur_param.c_str());
    char vb[32];
    std::snprintf(vb, sizeof(vb), "%.3f", vmax);
    dl->AddText(ImVec2(cp.x + 6, cp.y + 22), COL_TEXT_DIM, vb);
    std::snprintf(vb, sizeof(vb), "%.3f", vmin);
    dl->AddText(ImVec2(cp.x + 6, cp.y + CURVE_H - 18), COL_TEXT_DIM, vb);

    static const ImU32 comp_cols[4] = {IM_COL32(235, 100, 100, 255), IM_COL32(120, 220, 120, 255),
                                       IM_COL32(110, 160, 250, 255), IM_COL32(220, 200, 90, 255)};

    // Zero line, if it's in range.
    if (vmin < 0.0f && vmax > 0.0f) {
        float zy = v_to_y(0.0f);
        dl->AddLine(ImVec2(cx0, zy), ImVec2(cx0 + cvw, zy), COL_GRID);
    }

    ImGui::SetCursorScreenPos(cp);
    // Same AllowOverlap as the dope sheet's scrub surface: without it this
    // button claims hover for the whole rect and the key diamonds and tangent
    // handles submitted after it never receive the mouse.
    ImGui::SetNextItemAllowOverlap();
    ImGui::InvisibleButton("##curvebg", ImVec2(cw, CURVE_H));

    // The draw list clips to the window, not to this rect — spline overshoot
    // past the padded axis and steep Bezier tangent handles would paint over
    // whatever the caller lays out below the curve view.
    dl->PushClipRect(cp, ImVec2(cp.x + cw, cp.y + CURVE_H), true);

    bool curve_mutated = false;
    for (size_t ti = 0; ti < tl.tracks.size(); ti++) {
        Track& tr = tl.tracks[ti];
        if (tr.shader != cur_shader || tr.param != cur_param || tr.keys.empty()) continue;
        ImU32 col = comp_cols[std::clamp(tr.component, 0, 3)];

        // Sample at pixel resolution — eval() is the single source of truth,
        // so the drawn curve can't disagree with what playback produces.
        std::vector<ImVec2> pts;
        int steps = (int)std::min(cvw, 512.0f);
        pts.reserve(steps + 1);
        for (int s = 0; s <= steps; s++) {
            float t = ui.view_start + span * ((float)s / (float)steps);
            pts.push_back(ImVec2(ct_to_x(t), v_to_y(tr.eval(t))));
        }
        dl->AddPolyline(pts.data(), (int)pts.size(), col, 0, 1.6f);

        for (size_t k = 0; k < tr.keys.size(); k++) {
            Key& key = tr.keys[k];
            float x = ct_to_x(key.t), y = v_to_y(key.v);
            if (x < cx0 - 8 || x > cx0 + cvw + 8) continue;

            ImGui::PushID((int)(10000 + ti * 512 + k));
            ImGui::SetCursorScreenPos(ImVec2(x - KEY_R, y - KEY_R));
            ImGui::InvisibleButton("##ck", ImVec2(KEY_R * 2, KEY_R * 2));
            if (ImGui::IsItemActivated()) {
                ui.sel_camera = false;
                ui.sel_track = (int)ti;
                ui.sel_key = (int)k;
            }
            if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                key.t = std::clamp(snap(cx_to_t(ImGui::GetIO().MousePos.x)), 0.0f, tl.duration);
                // Clamp to the drawn axis, not the param's slider range. The
                // axis already spans the key extremes, so a key parked outside
                // the range stays where it is instead of being yanked to the
                // limit the first time you touch it — which matters now that
                // the range is user-editable and can be narrowed under
                // existing keys. (It's also why this isn't std::clamp on
                // min_val/max_val: those can be inverted, and std::clamp with
                // lo > hi is undefined behaviour.)
                key.v = std::clamp(y_to_v(ImGui::GetIO().MousePos.y), vmin, vmax);
                curve_mutated = true;
            }
            if (ImGui::IsItemDeactivatedAfterEdit()) {
                std::sort(tr.keys.begin(), tr.keys.end(),
                          [](const Key& a, const Key& b) { return a.t < b.t; });
            }
            if (ImGui::BeginPopupContextItem("##ckm2")) {
                static const char* comp_names[4] = {"x", "y", "z", "w"};
                ImGui::TextDisabled("%s @ %.3fs = %.4f",
                                    comp_names[std::clamp(tr.component, 0, 3)], key.t, key.v);
                ImGui::Separator();
                Interp mode = key.interp;
                if (interp_menu(mode)) {
                    // A multi-component param draws one diamond per channel,
                    // and channels sharing a value stack them pixel-coincident
                    // — which one a right-click lands on is submission order,
                    // not intent. So a type change applies to every channel's
                    // key at this time, the same scope as the dope sheet group
                    // row; per-channel types remain reachable through the
                    // expanded component rows.
                    for (auto& otr : tl.tracks) {
                        if (otr.shader != tr.shader || otr.param != tr.param) continue;
                        for (auto& okey : otr.keys) {
                            if (std::fabs(okey.t - key.t) < ft * 0.25f) okey.interp = mode;
                        }
                    }
                    curve_mutated = true;
                }
                if (ImGui::MenuItem("Delete")) {
                    tr.erase_at((int)k);
                    ui.sel_key = -1;
                    curve_mutated = true;
                    ImGui::EndPopup();
                    ImGui::PopID();
                    break;
                }
                ImGui::EndPopup();
            }
            bool sel = !ui.sel_camera && ui.sel_track == (int)ti && ui.sel_key == (int)k;
            draw_diamond(dl, ImVec2(x, y), KEY_R, sel ? COL_KEY_SEL : col, COL_KEY_EDGE);
            ImGui::PopID();

            // Bezier tangent handles, drawn a fixed pixel distance out so the
            // grab targets stay usable at any zoom.
            if (key.interp == Interp::Bezier || (k > 0 && tr.keys[k - 1].interp == Interp::Bezier)) {
                const float HL = 34.0f;
                float sec_per_px = span / cvw;
                float val_per_px = (vmax - vmin) / CURVE_H;
                struct Handle { float* tan; float dir; const char* id; };
                Handle hs[2] = {{&key.in_tan, -1.0f, "##ti"}, {&key.out_tan, 1.0f, "##to"}};
                for (int hi = 0; hi < 2; hi++) {
                    bool live = hi == 0 ? (k > 0 && tr.keys[k - 1].interp == Interp::Bezier)
                                        : (key.interp == Interp::Bezier);
                    if (!live) continue;
                    float hx = x + hs[hi].dir * HL;
                    float hy = y - (*hs[hi].tan) * (HL * sec_per_px) / val_per_px;
                    dl->AddLine(ImVec2(x, y), ImVec2(hx, hy), IM_COL32(180, 180, 190, 160), 1.0f);
                    dl->AddCircleFilled(ImVec2(hx, hy), 3.5f, IM_COL32(200, 200, 210, 220));

                    ImGui::PushID((int)(20000 + ti * 512 + k) + hi * 100000);
                    ImGui::SetCursorScreenPos(ImVec2(hx - 5, hy - 5));
                    ImGui::InvisibleButton(hs[hi].id, ImVec2(10, 10));
                    if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                        float dy = y - ImGui::GetIO().MousePos.y;
                        *hs[hi].tan = dy * val_per_px / (HL * sec_per_px);
                        curve_mutated = true;
                    }
                    ImGui::PopID();
                }
            }
        }
    }

    // Playhead over the curves.
    {
        float px = ct_to_x(tl.playhead);
        if (px >= cx0 && px <= cx0 + cvw) {
            dl->AddLine(ImVec2(px, cp.y), ImVec2(px, cp.y + CURVE_H), COL_PLAYHEAD, 1.5f);
        }
    }
    dl->PopClipRect();
    // Outside the clip so the border's edge pixels aren't shaved off.
    dl->AddRect(cp, ImVec2(cp.x + cw, cp.y + CURVE_H), COL_GRID);

    // Every key and tangent handle above moved the layout cursor to wherever
    // that button sat inside the rect. Park it on the rect's bottom edge, or
    // the caller's next widgets are laid out mid-curve, over the drawing.
    ImGui::SetCursorScreenPos(ImVec2(cp.x, cp.y + CURVE_H));

    if (curve_mutated) tl.revision++;
}
