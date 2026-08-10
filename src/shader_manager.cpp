#include "shader_manager.h"
#include <algorithm>
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
    : backend_(backend), shader_dir_(shader_dir) {
    common_timestamp_ = file_timestamp(shader_dir_ + "/common.metal");
}

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
    // common.metal is prepended to every shader, so a change to it reloads all
    uint64_t common_ts = file_timestamp(shader_dir_ + "/common.metal");
    bool common_changed = (common_ts != common_timestamp_ && common_ts != 0);
    common_timestamp_ = common_ts;

    bool any_reloaded = false;
    for (auto& entry : entries_) {
        uint64_t ts = file_timestamp(entry.full_path);
        if ((ts != entry.last_modified && ts != 0) || common_changed) {
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
            if (np.name != op.name || np.type != op.type) continue;
            std::memcpy(np.current_val, op.current_val, sizeof(np.current_val));

            // Carry a user's range override across the reload — but only while
            // the declaration it was made against still says the same thing.
            // Editing the `@param` line is how you'd expect to change a range,
            // and a sticky override would silently swallow that edit.
            bool decl_unchanged = np.decl_min[0] == op.decl_min[0] &&
                                  np.decl_max[0] == op.decl_max[0];
            if (has_range_override(op) && decl_unchanged) {
                std::memcpy(np.min_val, op.min_val, sizeof(np.min_val));
                std::memcpy(np.max_val, op.max_val, sizeof(np.max_val));
                np.logarithmic = op.logarithmic;
            }
            break;
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
// Format: // @param <name> <type> <values...> [@group <Section>] [@if <ctrl>=<v1>,<v2> ...]

// Parses the trailing annotations, in any order. `@if` clauses are
// `<name>=<v1>,<v2>` or `<name>!=<v1>`; the marker may be its own token or
// glued to the expression; multiple clauses are ANDed. `@group <name>` names
// the panel section the param is drawn under — display-only, so it never
// disturbs slot order. Malformed clauses are dropped, which leaves the param
// visible and ungrouped.
static void parse_annotations(const std::string& text, ShaderParam& p) {
    std::istringstream ss(text);
    std::string tok;
    while (ss >> tok) {
        if (tok.rfind("@group", 0) == 0) {
            tok = tok.substr(6);
            if (tok.empty() && !(ss >> tok)) break;
            p.group = tok;
            continue;
        }
        if (tok.rfind("@if", 0) == 0) {
            tok = tok.substr(3);
            if (tok.empty() && !(ss >> tok)) break;
        }

        size_t eq = tok.find('=');
        if (eq == std::string::npos) continue;

        ShaderParam::Condition c;
        size_t name_end = eq;
        if (name_end > 0 && tok[name_end - 1] == '!') {
            c.negate = true;
            name_end--;
        }
        c.name = tok.substr(0, name_end);

        std::string vals = tok.substr(eq + 1);
        for (size_t start = 0; start <= vals.size(); ) {
            size_t comma = vals.find(',', start);
            if (comma == std::string::npos) comma = vals.size();
            if (comma > start) c.values.push_back(vals.substr(start, comma - start));
            start = comma + 1;
        }

        if (!c.name.empty() && !c.values.empty()) p.conditions.push_back(std::move(c));
    }
}

std::vector<ShaderParam> ShaderManager::parse_params(const std::string& source) {
    std::vector<ShaderParam> params;
    std::istringstream stream(source);
    std::string line;

    while (std::getline(stream, line)) {
        // Find "// @param"
        size_t pos = line.find("// @param");
        if (pos == std::string::npos) continue;

        std::string rest = line.substr(pos + 9); // skip "// @param"

        // Split the annotations off first — enum eats every remaining token
        // as a label, so `@group`/`@if` have to be gone before values are
        // read. npos from both finds means no annotations (min keeps npos).
        ShaderParam p;
        size_t cut = std::min(rest.find("@if"), rest.find("@group"));
        if (cut != std::string::npos) {
            parse_annotations(rest.substr(cut), p);
            rest.resize(cut);
        }

        std::istringstream ls(rest);
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
        } else if (type_str == "enum") {
            p.type = ShaderParam::Enum;
            p.component_count = 1;
            std::string label;
            while (ls >> label) p.labels.push_back(label);
            if (p.labels.empty()) continue;
            p.min_val[0] = 0;
            p.max_val[0] = (float)(p.labels.size() - 1);
            p.default_val[0] = 0;
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

        // The parsed range is both the live range and the declared reference
        // the range popup resets to.
        std::memcpy(p.decl_min, p.min_val, sizeof(p.decl_min));
        std::memcpy(p.decl_max, p.max_val, sizeof(p.decl_max));

        params.push_back(std::move(p));
    }

    return params;
}
