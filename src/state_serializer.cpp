#include "state_serializer.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <cstring>

using json = nlohmann::json;

StateSerializer::StateSerializer(const std::string& file_path)
    : file_path_(file_path) {}

// ---- Save ----

void StateSerializer::save(const Camera& camera, const ShaderManager& shaders) {
    json j;

    // Camera
    j["camera"] = {
        {"pos", {camera.pos[0], camera.pos[1], camera.pos[2]}},
        {"target", {camera.target[0], camera.target[1], camera.target[2]}},
        {"fov", camera.fov},
        {"show_grid", camera.show_grid},
        {"rotate_speed", camera.rotate_speed},
        {"pan_speed", camera.pan_speed},
        {"zoom_speed", camera.zoom_speed},
        {"keyboard_speed", camera.keyboard_speed}
    };

    // Shader params — keyed by filename, then by param name
    json shaders_j;
    for (const auto& entry : shaders.entries()) {
        json params_j = json::object();
        for (const auto& p : entry.params) {
            params_j[p.name] = {p.current_val[0], p.current_val[1],
                                p.current_val[2], p.current_val[3]};
        }
        shaders_j[entry.filename] = params_j;
    }
    j["shaders"] = shaders_j;

    // Write atomically: write to tmp, then rename
    std::string tmp_path = file_path_ + ".tmp";
    std::ofstream f(tmp_path);
    if (!f.is_open()) {
        std::cerr << "[state] Failed to write " << tmp_path << "\n";
        return;
    }
    f << j.dump(2) << "\n";
    f.close();

    std::rename(tmp_path.c_str(), file_path_.c_str());
}

// ---- Load ----

static void load_float3(const json& j, const char* key, float* out) {
    if (j.contains(key) && j[key].is_array() && j[key].size() >= 3) {
        out[0] = j[key][0]; out[1] = j[key][1]; out[2] = j[key][2];
    }
}

void StateSerializer::load(Camera& camera, ShaderManager& shaders) {
    std::ifstream f(file_path_);
    if (!f.is_open()) {
        std::cout << "[state] No state file found, saving defaults\n";
        save(camera, shaders);
        snapshot(camera, shaders);
        return;
    }

    json j;
    try {
        f >> j;
    } catch (const json::parse_error& e) {
        std::cerr << "[state] Parse error: " << e.what() << "\n";
        snapshot(camera, shaders);
        return;
    }

    // Camera
    if (j.contains("camera")) {
        auto& c = j["camera"];
        load_float3(c, "pos", camera.pos);
        load_float3(c, "target", camera.target);
        if (c.contains("fov"))            camera.fov            = c["fov"];
        if (c.contains("show_grid"))      camera.show_grid      = c["show_grid"];
        if (c.contains("rotate_speed"))   camera.rotate_speed   = c["rotate_speed"];
        if (c.contains("pan_speed"))      camera.pan_speed      = c["pan_speed"];
        if (c.contains("zoom_speed"))     camera.zoom_speed     = c["zoom_speed"];
        if (c.contains("keyboard_speed")) camera.keyboard_speed = c["keyboard_speed"];
    }

    // Shader params
    if (j.contains("shaders") && j["shaders"].is_object()) {
        for (auto& [filename, params_j] : j["shaders"].items()) {
            auto& params = shaders.get_params_mut(filename);
            for (auto& p : params) {
                if (params_j.contains(p.name) && params_j[p.name].is_array() &&
                    params_j[p.name].size() >= 4) {
                    for (int i = 0; i < 4; i++) {
                        p.current_val[i] = params_j[p.name][i];
                    }
                }
            }
        }
    }

    std::cout << "[state] Loaded state from " << file_path_ << "\n";
    snapshot(camera, shaders);
}

// ---- Change detection ----

static bool camera_eq(const Camera& a, const Camera& b) {
    return std::memcmp(a.pos, b.pos, sizeof(a.pos)) == 0 &&
           std::memcmp(a.target, b.target, sizeof(a.target)) == 0 &&
           a.fov == b.fov &&
           a.show_grid == b.show_grid &&
           a.rotate_speed == b.rotate_speed &&
           a.pan_speed == b.pan_speed &&
           a.zoom_speed == b.zoom_speed &&
           a.keyboard_speed == b.keyboard_speed;
}

bool StateSerializer::state_differs(const Camera& camera, const ShaderManager& shaders) const {
    if (!camera_eq(camera, last_camera_)) return true;

    for (size_t si = 0; si < last_shader_params_.size(); si++) {
        const auto& entries = shaders.entries();
        if (si >= entries.size()) return true;
        const auto& last_params = last_shader_params_[si].second;
        const auto& curr_params = entries[si].params;
        if (last_params.size() != curr_params.size()) return true;
        for (size_t pi = 0; pi < curr_params.size(); pi++) {
            if (std::memcmp(curr_params[pi].current_val, last_params[pi].current_val,
                            sizeof(float) * 4) != 0) {
                return true;
            }
        }
    }
    return false;
}

void StateSerializer::snapshot(const Camera& camera, const ShaderManager& shaders) {
    last_camera_ = camera;
    last_shader_params_.clear();
    for (const auto& entry : shaders.entries()) {
        last_shader_params_.emplace_back(entry.filename, entry.params);
    }
}

void StateSerializer::save_if_changed(const Camera& camera, const ShaderManager& shaders, float time) {
    if (state_differs(camera, shaders)) {
        dirty_ = true;
        dirty_time_ = time;
        snapshot(camera, shaders);
    } else if (dirty_ && (time - dirty_time_) >= debounce_seconds_) {
        save(camera, shaders);
        dirty_ = false;
    }
}
