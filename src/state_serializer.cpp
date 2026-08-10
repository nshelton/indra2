#include "state_serializer.h"
#include "timeline.h"
#include <nlohmann/json.hpp>
#include <cmath>
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
    fields_.push_back({json_path, Field::Enum, p, std::move(labels), false});
}
void StateSerializer::session_bool_field(const char* json_path, bool* p) {
    fields_.push_back({json_path, Field::Bool, p, {}, true});
}

// ---- Save ----

void StateSerializer::save(const ShaderManager& shaders) {
    // The main state file is the session, so session-only fields belong in it.
    save_to_impl(file_path_, shaders, nullptr, true);
}

void StateSerializer::save_to(const std::string& path, const ShaderManager& shaders,
                              const Timeline* tl) {
    save_to_impl(path, shaders, tl, false);
}

void StateSerializer::save_to_impl(const std::string& path, const ShaderManager& shaders,
                                   const Timeline* tl, bool include_session) {
    json j;

    for (const auto& f : fields_) {
        if (f.session_only && !include_session) continue;
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

    // Slider range overrides live in their own block rather than beside the
    // values, so loading can be "reset every range to what the shader
    // declared, then apply this". That keeps loads idempotent: a preset that
    // doesn't mention a param cannot leave a stale override behind from
    // whatever was loaded before it. Only overridden params are written, so
    // files stay untouched until someone actually widens something.
    json ranges_j = json::object();
    for (const auto& entry : shaders.entries()) {
        json per_shader = json::object();
        for (const auto& p : entry.params) {
            if (!has_range_override(p)) continue;
            json r = json::object();
            r["min"] = p.min_val[0];
            r["max"] = p.max_val[0];
            if (p.logarithmic) r["log"] = true;
            // The declaration this override was made against. Editing the
            // `@param` line later has to win over a stale override, and
            // across a restart this record is the only way to notice.
            r["decl"] = {p.decl_min[0], p.decl_max[0]};
            per_shader[p.name] = r;
        }
        if (!per_shader.empty()) ranges_j[entry.filename] = per_shader;
    }
    j["param_ranges"] = ranges_j;

    // The animation, when this blob is a preset. Absent from the main state
    // file, where animations/_working.json is the session scratch instead.
    if (tl) tl->to_json(j["timeline"]);

    // Write atomically: write to tmp, then rename
    std::string tmp_path = path + ".tmp";
    std::ofstream f(tmp_path);
    if (!f.is_open()) {
        std::cerr << "[state] Failed to write " << tmp_path << "\n";
        return;
    }
    f << j.dump(2) << "\n";
    f.close();

    std::rename(tmp_path.c_str(), path.c_str());
}

// ---- Load ----

void StateSerializer::load(ShaderManager& shaders) {
    if (!load_from_impl(file_path_, shaders, nullptr, true)) {
        std::cout << "[state] No usable state file, saving defaults\n";
        save(shaders);
    }
    snapshot(shaders);
}

bool StateSerializer::load_from(const std::string& path, ShaderManager& shaders, Timeline* tl) {
    return load_from_impl(path, shaders, tl, false);
}

bool StateSerializer::load_from_impl(const std::string& path, ShaderManager& shaders,
                                     Timeline* tl, bool include_session) {
    std::ifstream f(path);
    if (!f.is_open()) return false;

    json j;
    try {
        f >> j;
    } catch (const json::parse_error& e) {
        std::cerr << "[state] Parse error in " << path << ": " << e.what() << "\n";
        return false;
    }

    for (const auto& f : fields_) {
        if (f.session_only && !include_session) continue;
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

    // Shader params. is_number() guards the element: nlohmann serializes a
    // NaN or Inf as `null`, and an unguarded assignment from null throws a
    // json::type_error that the try/catch above (which only wraps the parse)
    // would not catch.
    if (j.contains("shaders") && j["shaders"].is_object()) {
        for (auto& [filename, params_j] : j["shaders"].items()) {
            auto& params = shaders.get_params_mut(filename);
            for (auto& p : params) {
                if (params_j.contains(p.name) && params_j[p.name].is_array() &&
                    params_j[p.name].size() >= 4) {
                    for (int i = 0; i < 4; i++) {
                        const json& e = params_j[p.name][i];
                        if (e.is_number()) p.current_val[i] = e;
                    }
                }
            }
        }
    }

    // Range overrides are authoritative as a set — reset to the shader's
    // declared range first, unconditionally, then apply whatever the file
    // carries. See the matching comment in save_to.
    {
        std::vector<std::string> filenames;
        for (const auto& e : shaders.entries()) filenames.push_back(e.filename);
        for (const auto& fn : filenames) {
            for (auto& p : shaders.get_params_mut(fn)) {
                std::memcpy(p.min_val, p.decl_min, sizeof(p.min_val));
                std::memcpy(p.max_val, p.decl_max, sizeof(p.max_val));
                p.logarithmic = false;
            }
        }
    }
    if (j.contains("param_ranges") && j["param_ranges"].is_object()) {
        for (auto& [filename, per_shader] : j["param_ranges"].items()) {
            if (!per_shader.is_object()) continue;
            for (auto& p : shaders.get_params_mut(filename)) {
                if (!param_range_editable(p)) continue;
                if (!per_shader.contains(p.name) || !per_shader[p.name].is_object()) continue;
                const json& r = per_shader[p.name];
                // Drop an override recorded against a declaration that has
                // since changed — the shader edit is the newer intent. This
                // is the restart-time twin of the check in reload_shader.
                // A missing "decl" is an older file: accept it.
                if (r.contains("decl") && r["decl"].is_array() && r["decl"].size() >= 2 &&
                    r["decl"][0].is_number() && r["decl"][1].is_number()) {
                    float dmin = r["decl"][0], dmax = r["decl"][1];
                    if (dmin != p.decl_min[0] || dmax != p.decl_max[0]) continue;
                }
                if (r.contains("min") && r["min"].is_number()) {
                    float v = r["min"];
                    if (std::isfinite(v)) for (int c = 0; c < 4; c++) p.min_val[c] = v;
                }
                if (r.contains("max") && r["max"].is_number()) {
                    float v = r["max"];
                    if (std::isfinite(v)) for (int c = 0; c < 4; c++) p.max_val[c] = v;
                }
                if (r.contains("log") && r["log"].is_boolean()) p.logarithmic = r["log"];
            }
        }
    }

    // The animation. A preset is authoritative for the whole state, so a file
    // with no "timeline" block clears the timeline rather than leaving the
    // previous animation to drive parameters it was never authored against.
    // That covers every preset saved before animations existed.
    if (tl) {
        if (j.contains("timeline")) tl->from_json(j["timeline"]);
        else                        tl->clear();
    }

    std::cout << "[state] Loaded state from " << path << "\n";
    return true;
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
            const auto& c = curr_params[pi];
            const auto& l = last_params[pi];
            if (std::memcmp(c.current_val, l.current_val, sizeof(float) * 4) != 0) {
                return true;
            }
            // Range overrides are persisted too, so a range-only edit has to
            // mark the state dirty — otherwise it's silently lost on quit.
            if (std::memcmp(c.min_val, l.min_val, sizeof(float) * 4) != 0 ||
                std::memcmp(c.max_val, l.max_val, sizeof(float) * 4) != 0 ||
                c.logarithmic != l.logarithmic) {
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
