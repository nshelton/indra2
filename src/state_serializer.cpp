#include "state_serializer.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <cstring>

using json = nlohmann::json;
using jptr = json::json_pointer;

StateSerializer::StateSerializer(const std::string& file_path)
    : file_path_(file_path) {}

// ---- Field registration ----

void StateSerializer::float_field(const char* json_path, float* p) {
    fields_.push_back({json_path, Field::Float, p, {}});
}
void StateSerializer::int_field(const char* json_path, int* p) {
    fields_.push_back({json_path, Field::Int, p, {}});
}
void StateSerializer::bool_field(const char* json_path, bool* p) {
    fields_.push_back({json_path, Field::Bool, p, {}});
}
void StateSerializer::float3_field(const char* json_path, float* p) {
    fields_.push_back({json_path, Field::Float3, p, {}});
}
void StateSerializer::enum_field(const char* json_path, int* p, std::vector<std::string> labels) {
    fields_.push_back({json_path, Field::Enum, p, std::move(labels)});
}

// ---- Save ----

void StateSerializer::save(const ShaderManager& shaders) {
    json j;

    for (const auto& f : fields_) {
        jptr path(f.path);
        switch (f.type) {
            case Field::Float:  j[path] = *(float*)f.ptr; break;
            case Field::Int:    j[path] = *(int*)f.ptr; break;
            case Field::Bool:   j[path] = *(bool*)f.ptr; break;
            case Field::Float3: {
                float* v = (float*)f.ptr;
                j[path] = {v[0], v[1], v[2]};
                break;
            }
            case Field::Enum: {
                int i = *(int*)f.ptr;
                if (i >= 0 && i < (int)f.labels.size()) j[path] = f.labels[i];
                break;
            }
        }
    }

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

void StateSerializer::load(ShaderManager& shaders) {
    std::ifstream f(file_path_);
    if (!f.is_open()) {
        std::cout << "[state] No state file found, saving defaults\n";
        save(shaders);
        snapshot(shaders);
        return;
    }

    json j;
    try {
        f >> j;
    } catch (const json::parse_error& e) {
        std::cerr << "[state] Parse error: " << e.what() << "\n";
        snapshot(shaders);
        return;
    }

    for (const auto& f : fields_) {
        jptr path(f.path);
        if (!j.contains(path)) continue;
        const json& v = j[path];
        switch (f.type) {
            case Field::Float:  if (v.is_number()) *(float*)f.ptr = v; break;
            case Field::Int:    if (v.is_number()) *(int*)f.ptr = v; break;
            case Field::Bool:   if (v.is_boolean()) *(bool*)f.ptr = v; break;
            case Field::Float3:
                if (v.is_array() && v.size() >= 3) {
                    float* out = (float*)f.ptr;
                    out[0] = v[0]; out[1] = v[1]; out[2] = v[2];
                }
                break;
            case Field::Enum:
                if (v.is_string()) {
                    for (int i = 0; i < (int)f.labels.size(); i++) {
                        if (v == f.labels[i]) { *(int*)f.ptr = i; break; }
                    }
                }
                break;
        }
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
    snapshot(shaders);
}

// ---- Change detection ----

std::array<uint8_t, 16> StateSerializer::field_bytes(const Field& f) const {
    std::array<uint8_t, 16> b = {};
    switch (f.type) {
        case Field::Float:  std::memcpy(b.data(), f.ptr, sizeof(float)); break;
        case Field::Int:
        case Field::Enum:   std::memcpy(b.data(), f.ptr, sizeof(int)); break;
        case Field::Bool:   std::memcpy(b.data(), f.ptr, sizeof(bool)); break;
        case Field::Float3: std::memcpy(b.data(), f.ptr, sizeof(float) * 3); break;
    }
    return b;
}

bool StateSerializer::state_differs(const ShaderManager& shaders) const {
    for (size_t i = 0; i < fields_.size(); i++) {
        if (field_bytes(fields_[i]) != last_field_bytes_[i]) return true;
    }

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

void StateSerializer::snapshot(const ShaderManager& shaders) {
    last_field_bytes_.clear();
    for (const auto& f : fields_) last_field_bytes_.push_back(field_bytes(f));

    last_shader_params_.clear();
    for (const auto& entry : shaders.entries()) {
        last_shader_params_.emplace_back(entry.filename, entry.params);
    }
}

void StateSerializer::save_if_changed(const ShaderManager& shaders, float time) {
    if (state_differs(shaders)) {
        dirty_ = true;
        dirty_time_ = time;
        snapshot(shaders);
    } else if (dirty_ && (time - dirty_time_) >= debounce_seconds_) {
        save(shaders);
        dirty_ = false;
    }
}
