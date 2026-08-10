#pragma once
#include "types.h"
#include "gui.h"
// Forward-declaration header only — keeps the full nlohmann include out of
// everything that touches a Timeline.
#include <nlohmann/json_fwd.hpp>
#include <string>
#include <vector>

class ShaderManager;

// ---- Animation data model ----
//
// Every tunable in the engine is already a named ShaderParam holding a
// float[4] (types.h), so a timeline needs no reflection layer: a track is
// keyed by (shader filename, param name, component). Never by slot index —
// slots shift when a `@param` line is inserted mid-list, and state.json keys
// on name for exactly the same reason.

enum class Interp : int { Step = 0, Linear = 1, Smooth = 2, Bezier = 3 };

const char* interp_name(Interp i);

struct Key {
    float  t = 0.0f;
    float  v = 0.0f;
    Interp interp = Interp::Smooth;
    float  in_tan = 0.0f, out_tan = 0.0f;  // Bezier only, value-units per second
    // Transient UI selection. Lives on the key so sorts and inserts can't
    // orphan it the way stored indices would. Never serialized.
    bool   selected = false;
};

// One scalar channel. A float3 param owns three of these.
struct Track {
    std::string shader;      // "raymarch.metal" | "pathtrace.metal" | ...
    std::string param;       // ShaderParam::name
    int  component = 0;      // 0..3
    bool enabled = true;
    std::vector<Key> keys;   // kept sorted by t

    float eval(float t) const;
    // Insert or replace a key. `merge_window` is how close in time counts as
    // the same key (callers pass half a frame). Returns the key's index.
    int   insert(float t, float v, float merge_window, Interp interp = Interp::Smooth);
    void  erase_at(int index);
};

// The camera is one row, not eight tracks: slerp and log-distance lerp are
// joint operations across components and can't be decomposed into independent
// scalar curves.
struct CameraKey {
    float  t = 0.0f;
    float  target[3] = {0, 0, 0};
    float  dir[3]    = {0, 0, 1};  // normalize(pos - target)
    float  log_dist  = 0.0f;       // log(|pos - target|)
    float  fov       = 1.2f;
    Interp interp    = Interp::Smooth;
    bool   selected  = false;      // transient UI selection, never serialized
};

struct Timeline {
    // Master switch. Off means fully inert, not just hidden: apply() does
    // nothing, the camera stays manual, and the param sliders lose their
    // keying affordances. Keys are preserved, so it is a safe thing to flip
    // while noodling. Persisted in state.json, not in the .anim.json.
    bool  enabled = true;

    float duration = 10.0f;
    float fps      = 30.0f;
    float playhead = 0.0f;
    bool  playing      = false;
    bool  loop         = true;
    bool  auto_key     = false;
    bool  drive_camera = true;

    std::vector<Track>     tracks;
    std::vector<CameraKey> cam_keys;

    // Bumped by every mutation; the autosave compares against what it last
    // wrote instead of memcmp'ing a variable-length structure.
    uint64_t revision = 0;

    // Set by the param UI while a slider is held. apply() leaves that channel
    // alone next frame, otherwise the curve stomps the drag before the user
    // ever sees it move. Cleared by the host each frame before drawing the UI.
    std::string editing_shader, editing_param;
    // Same latch for the Camera panel's pose widgets: with auto-key on they
    // are enabled while the timeline drives the camera, and without this
    // eval_camera overwrites every edit (fov most visibly — it has no mouse
    // path, so the slider is its only input) before it ever renders.
    bool editing_camera = false;

    // --- Queries ---
    const Track* find(const std::string& shader, const std::string& param, int comp) const;
    Track*       find(const std::string& shader, const std::string& param, int comp);
    bool         has_track(const std::string& shader, const std::string& param) const;
    // Any un-muted track for the param. has_track && !track_active = fully
    // muted: keys are parked and the slider owns the value.
    bool         track_active(const std::string& shader, const std::string& param) const;
    bool         camera_driven() const { return enabled && drive_camera && !cam_keys.empty(); }
    float        frame_time() const { return fps > 0.0f ? 1.0f / fps : 1.0f / 30.0f; }

    // --- Mutation ---
    Track& get_or_create(const std::string& shader, const ShaderParam& p, int comp);
    void   key_param(float t, const std::string& shader, const ShaderParam& p);
    void   remove_param(const std::string& shader, const std::string& param);
    void   key_camera(float t, const Camera& cam);
    void   remove_camera_key(int index);
    void   clear();

    // --- Evaluation ---
    // Writes into ShaderParam::current_val and Camera, so every downstream
    // system (sliders, the params_changed memcmp, serialization) sees animated
    // values through the paths it already uses.
    void   apply(float t, ShaderManager& shaders, Camera& cam) const;
    void   eval_camera(float t, Camera& cam) const;
    // The interpolated channels (target/dir/log_dist/fov) at t, before they
    // compose into a pos. The camera curve editor plots exactly these, so
    // what it shows is what playback computes — slerp kinks included.
    void   eval_camera_channels(float t, CameraKey& out) const;

    // --- Serialization ---
    // to_json/from_json are the reusable value: the same block is written
    // standalone into animations/ and embedded under "timeline" in a preset.
    // `enabled` is never part of it — see the field's comment.
    void to_json(nlohmann::json& j) const;
    void from_json(const nlohmann::json& j);
    bool save(const std::string& path) const;
    bool load(const std::string& path);
};

// ---- Timeline editor UI state ----
// Lives across frames; owned by main.cpp and passed into draw_timeline.
struct TimelineUI {
    float view_start = 0.0f;     // visible time range (zoom/pan)
    float view_end   = 10.0f;
    // What a full-span view_end equals. While the view spans [0, duration]
    // exactly it follows duration edits; zoom/pan detaches it, and zooming
    // fully back out (clamped to the animation) reattaches. -1 = fit on
    // first draw, so a loaded animation opens at its own length.
    float fitted_duration = -1.0f;
    bool  show_curves = false;

    // Last-clicked key (drives the curve editor's focus). The multi-key
    // selection itself lives as `selected` flags on the keys.
    bool  sel_camera = false;
    int   sel_track  = -1;
    int   sel_key    = -1;

    // Marquee drag state (screen coords) and the rigid multi-key move
    // accumulator — see timeline_ui.cpp.
    bool  marquee = false;
    float marquee_x0 = 0.0f, marquee_y0 = 0.0f;
    float drag_accum = 0.0f, drag_applied = 0.0f;

    // Camera curve editor: visibility bitmask over the 8 channels
    // (tgt.xyz, dir.xyz, log_dist, fov).
    unsigned cam_mask = 0xFFu;

    // Copy/paste clipboard: the selected keys, times stored relative to the
    // earliest one. Param keys carry their track identity and re-target by
    // (shader, param, component) on paste; paste lands the earliest key on
    // the playhead. App-local, not the OS clipboard.
    struct ClipKey {
        std::string shader, param;
        int component = 0;
        Key key;
    };
    std::vector<ClipKey>   clip_keys;
    std::vector<CameraKey> clip_cam;

    // Groups expanded to their per-component rows, keyed "shader|param".
    std::vector<std::string> expanded;

    bool is_expanded(const std::string& key) const;
    void toggle_expanded(const std::string& key);
};

// The dope sheet, transport and curve editor. Contents only — the caller owns
// the ImGui::Begin/End so the window can be docked or embedded.
void draw_timeline(Timeline& tl, TimelineUI& ui, ShaderManager& shaders, Camera& cam);
