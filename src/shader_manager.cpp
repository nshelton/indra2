#include "shader_manager.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>

namespace fs = std::filesystem;

const std::vector<ShaderParam> ShaderManager::empty_params_;
const std::string ShaderManager::empty_string_;

static std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return "";
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static uint64_t file_timestamp(const std::string& path) {
    std::error_code ec;
    auto ftime = fs::last_write_time(path, ec);
    if (ec) return 0;
    return (uint64_t)ftime.time_since_epoch().count();
}

ShaderManager::ShaderManager(MetalBackend& backend, const std::string& shader_dir)
    : backend_(backend), shader_dir_(shader_dir) {}

void ShaderManager::register_shader(const std::string& filename, const std::string& kernel_name) {
    ShaderEntry entry;
    entry.filename = filename;
    entry.kernel_name = kernel_name;
    entry.full_path = shader_dir_ + "/" + filename;
    entries_.push_back(std::move(entry));

    // Initial load
    reload_shader(entries_.back());
}

bool ShaderManager::poll_and_reload() {
    bool any_reloaded = false;
    for (auto& entry : entries_) {
        uint64_t ts = file_timestamp(entry.full_path);
        if (ts != entry.last_modified && ts != 0) {
            reload_shader(entry);
            any_reloaded = true;
        }
    }
    return any_reloaded;
}

void ShaderManager::reload_shader(ShaderEntry& entry) {
    std::string common_src = read_file(shader_dir_ + "/common.metal");
    std::string shader_src = read_file(entry.full_path);

    if (shader_src.empty()) {
        entry.error = "Could not read file: " + entry.full_path;
        return;
    }

    // Parse params from shader source only (not common)
    auto new_params = parse_params(shader_src);

    // Preserve current_val from old params if names match
    for (auto& np : new_params) {
        for (const auto& op : entry.params) {
            if (np.name == op.name && np.type == op.type) {
                std::memcpy(np.current_val, op.current_val, sizeof(np.current_val));
                break;
            }
        }
    }
    entry.params = std::move(new_params);

    // Combine sources with #line directive for correct error reporting
    std::string combined = common_src + "\n#line 1\n" + shader_src;

    std::string err;
    int new_pipeline = backend_.compile_kernel(combined, entry.kernel_name, err);
    if (new_pipeline >= 0) {
        entry.pipeline_id = new_pipeline;
        entry.error.clear();
        std::cout << "[shader] Compiled " << entry.filename << " successfully (" << entry.params.size() << " params)\n";
    } else {
        entry.error = err;
        std::cerr << "[shader] Error in " << entry.filename << ": " << err << "\n";
        // Keep old pipeline_id
    }

    entry.last_modified = file_timestamp(entry.full_path);
}

int ShaderManager::get_pipeline(const std::string& kernel_name) const {
    for (const auto& e : entries_) {
        if (e.kernel_name == kernel_name) return e.pipeline_id;
    }
    return -1;
}

const std::vector<ShaderParam>& ShaderManager::get_params(const std::string& filename) const {
    for (const auto& e : entries_) {
        if (e.filename == filename) return e.params;
    }
    return empty_params_;
}

std::vector<ShaderParam>& ShaderManager::get_params_mut(const std::string& filename) {
    for (auto& e : entries_) {
        if (e.filename == filename) return e.params;
    }
    // Should never happen in practice
    static std::vector<ShaderParam> dummy;
    return dummy;
}

const std::string& ShaderManager::get_error(const std::string& filename) const {
    for (const auto& e : entries_) {
        if (e.filename == filename) return e.error;
    }
    return empty_string_;
}

// ---- Param parsing ----
// Format: // @param <name> <type> <values...>

std::vector<ShaderParam> ShaderManager::parse_params(const std::string& source) {
    std::vector<ShaderParam> params;
    std::istringstream stream(source);
    std::string line;

    while (std::getline(stream, line)) {
        // Find "// @param"
        size_t pos = line.find("// @param");
        if (pos == std::string::npos) continue;

        std::string rest = line.substr(pos + 9); // skip "// @param"
        std::istringstream ls(rest);

        ShaderParam p;
        std::string type_str;
        ls >> p.name >> type_str;

        if (p.name.empty() || type_str.empty()) continue;

        p.is_color = false;

        if (type_str == "float") {
            p.type = ShaderParam::Float;
            p.component_count = 1;
            ls >> p.min_val[0] >> p.max_val[0] >> p.default_val[0];
        } else if (type_str == "int") {
            p.type = ShaderParam::Int;
            p.component_count = 1;
            ls >> p.min_val[0] >> p.max_val[0] >> p.default_val[0];
        } else if (type_str == "float2") {
            p.type = ShaderParam::Float2;
            p.component_count = 2;
            for (int i = 0; i < 2; i++) ls >> p.min_val[i];
            for (int i = 0; i < 2; i++) ls >> p.max_val[i];
            for (int i = 0; i < 2; i++) ls >> p.default_val[i];
        } else if (type_str == "float3" || type_str == "color3") {
            p.type = ShaderParam::Float3;
            p.component_count = 3;
            p.is_color = (type_str == "color3");
            for (int i = 0; i < 3; i++) ls >> p.min_val[i];
            for (int i = 0; i < 3; i++) ls >> p.max_val[i];
            for (int i = 0; i < 3; i++) ls >> p.default_val[i];
        } else if (type_str == "float4" || type_str == "color4") {
            p.type = ShaderParam::Float4;
            p.component_count = 4;
            p.is_color = (type_str == "color4");
            for (int i = 0; i < 4; i++) ls >> p.min_val[i];
            for (int i = 0; i < 4; i++) ls >> p.max_val[i];
            for (int i = 0; i < 4; i++) ls >> p.default_val[i];
        } else {
            continue; // unknown type
        }

        // Initialize current_val to default
        std::memcpy(p.current_val, p.default_val, sizeof(p.current_val));

        params.push_back(std::move(p));
    }

    return params;
}
