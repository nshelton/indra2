#pragma once
#include "types.h"
#include "gui.h"
#include "shader_manager.h"
#include <string>

class StateSerializer {
public:
    StateSerializer(const std::string& file_path);

    // Load state from disk. Applies values to camera and shader params.
    // Missing keys are silently ignored (keeps defaults).
    void load(Camera& camera, ShaderManager& shaders);

    // Check if state changed; saves after 2s of no changes. Call once per frame.
    void save_if_changed(const Camera& camera, const ShaderManager& shaders, float time);

private:
    std::string file_path_;

    // Snapshot of last-compared state for change detection
    Camera last_camera_;
    std::vector<std::pair<std::string, std::vector<ShaderParam>>> last_shader_params_;

    // Debounce: save 2 seconds after the last change
    bool dirty_ = false;
    float dirty_time_ = 0;
    static constexpr float debounce_seconds_ = 2.0f;

    bool state_differs(const Camera& camera, const ShaderManager& shaders) const;
    void save(const Camera& camera, const ShaderManager& shaders);
    void snapshot(const Camera& camera, const ShaderManager& shaders);
};
