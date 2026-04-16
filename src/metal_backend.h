#pragma once
#include "types.h"
#include <memory>
#include <string>
#include <vector>

struct SDL_Window;

class MetalBackend {
public:
    MetalBackend();
    ~MetalBackend();

    bool init(SDL_Window* window);
    void shutdown();

    // Texture management — returns opaque ID, -1 on failure
    int create_texture(const TextureDesc& desc);
    void destroy_texture(int texture_id);
    void resize_texture(int texture_id, uint32_t width, uint32_t height);

    // Buffer management — returns opaque ID, -1 on failure
    int create_buffer(size_t size_bytes, const std::string& label);
    void destroy_buffer(int buffer_id);
    void write_buffer(int buffer_id, const void* data, size_t size, size_t offset = 0);

    // Shader compilation — returns pipeline ID, -1 on failure
    int compile_kernel(const std::string& msl_source,
                       const std::string& kernel_function_name,
                       std::string& error_out);

    // Dispatch
    struct DispatchParams {
        int pipeline_id;
        uint32_t grid_width;
        uint32_t grid_height;
        uint32_t threadgroup_w = 16;
        uint32_t threadgroup_h = 16;
        std::vector<int> textures;
        std::vector<int> buffers;
    };

    // ImGui lifecycle (wraps ObjC ImGui Metal backend)
    void imgui_init();
    void imgui_shutdown();
    void imgui_new_frame();

    void begin_frame();
    void dispatch(const DispatchParams& params);
    void blit_to_screen(int source_texture_id);
    void render_imgui();
    void end_frame();

    uint32_t drawable_width() const;
    uint32_t drawable_height() const;

    // Type-erased Metal pointers (for external use if needed)
    void* raw_device() const;
    void* raw_command_queue() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
