#pragma once
#include "types.h"
#include "metal_backend.h"
#include <string>
#include <vector>

class ShaderManager {
public:
    ShaderManager(MetalBackend& backend, const std::string& shader_dir);

    // Call once per frame. Returns true if any shader was recompiled.
    bool poll_and_reload();

    // Get current pipeline ID for a kernel. -1 if not loaded or failed.
    int get_pipeline(const std::string& kernel_name) const;

    // Get parsed params for a shader file (by filename, e.g. "raymarch.metal")
    const std::vector<ShaderParam>& get_params(const std::string& filename) const;

    // Mutable access for GUI to modify current_val
    std::vector<ShaderParam>& get_params_mut(const std::string& filename);

    // Get last error for a shader file. Empty if no error.
    const std::string& get_error(const std::string& filename) const;

    // Register a shader file and its kernel function name.
    void register_shader(const std::string& filename, const std::string& kernel_name);

    struct ShaderEntry {
        std::string filename;
        std::string kernel_name;
        std::string full_path;
        int pipeline_id = -1;
        uint64_t last_modified = 0;
        std::vector<ShaderParam> params;
        std::string error;
    };

    const std::vector<ShaderEntry>& entries() const { return entries_; }

private:
    MetalBackend& backend_;
    std::string shader_dir_;
    std::vector<ShaderEntry> entries_;

    static const std::vector<ShaderParam> empty_params_;
    static const std::string empty_string_;

    void reload_shader(ShaderEntry& entry);
    std::vector<ShaderParam> parse_params(const std::string& source);
};
