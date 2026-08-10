#pragma once
#include "types.h"
#include "shader_manager.h"
#include <array>
#include <string>
#include <vector>

struct Timeline;  // timeline.h — one-way, so timeline.cpp stays serializer-free

// Persists registered app values + all shader params to a JSON file.
// Register fields once before load(); save/load/change-detection all walk
// the same table, so a registered field can never be half-persisted.
class StateSerializer {
public:
    StateSerializer(const std::string& file_path);

    // json_path is a JSON pointer, e.g. "/camera/pos" — nesting is created on save.
    void float_field(const char* json_path, float* p);
    void int_field(const char* json_path, int* p);
    void bool_field(const char* json_path, bool* p);
    void float3_field(const char* json_path, float* p);
    // Saved as a string label; loaded by label lookup. *p indexes labels.
    void enum_field(const char* json_path, int* p, std::vector<std::string> labels);
    // Belongs to the session, not to a preset: persisted to the main state
    // file but skipped entirely by save_to/load_from. For app preferences
    // that would be surprising to inherit from someone else's preset.
    void session_bool_field(const char* json_path, bool* p);

    // Load state from disk into registered fields and shader params.
    // Missing keys are silently ignored (keeps defaults).
    void load(ShaderManager& shaders);

    // Check if state changed; saves after 2s of no changes. Call once per frame.
    void save_if_changed(const ShaderManager& shaders, float time);

    // Unconditional save — call on shutdown so debounced changes aren't lost.
    void save(const ShaderManager& shaders);

    // Presets: same full-state blob, arbitrary path, plus the animation. A
    // preset owns its timeline — camera keys are absolute positions inside one
    // specific fractal, so an animation and the parameters it was authored
    // against are a single artifact. Passing tl on load makes a file with no
    // "timeline" block CLEAR the timeline, which is what makes every preset
    // authoritative for the whole state.
    //
    // load_from does not snapshot, so the autosave sees the loaded state as a
    // change and persists it to the main state file.
    void save_to(const std::string& path, const ShaderManager& shaders,
                 const Timeline* tl = nullptr);
    bool load_from(const std::string& path, ShaderManager& shaders,
                   Timeline* tl = nullptr);

private:
    struct Field {
        std::string path;
        enum Type { Float, Int, Bool, Float3, Enum } type;
        void* ptr;
        std::vector<std::string> labels;
        bool session_only = false;
    };

    std::string file_path_;
    std::vector<Field> fields_;

    // Snapshot of last-compared state for change detection
    std::vector<std::array<uint8_t, 16>> last_field_bytes_;
    std::vector<std::pair<std::string, std::vector<ShaderParam>>> last_shader_params_;

    // Debounce: save 2 seconds after the last change
    bool dirty_ = false;
    float dirty_time_ = 0;
    static constexpr float debounce_seconds_ = 2.0f;

    // Shared bodies. include_session is true only for the main state file:
    // presets must neither write nor apply session-only fields.
    void save_to_impl(const std::string& path, const ShaderManager& shaders,
                      const Timeline* tl, bool include_session);
    bool load_from_impl(const std::string& path, ShaderManager& shaders,
                        Timeline* tl, bool include_session);

    std::array<uint8_t, 16> field_bytes(const Field& f) const;
    bool state_differs(const ShaderManager& shaders) const;
    void snapshot(const ShaderManager& shaders);
};
