#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <cstring>

// Parsed from shader comment headers
struct ShaderParam {
    enum Type { Float, Int, Float2, Float3, Float4 };

    std::string name;
    Type type;
    bool is_color = false;       // true for color3/color4 — renders as ColorEdit
    float min_val[4]     = {};
    float max_val[4]     = {};
    float default_val[4] = {};
    float current_val[4] = {};
    int component_count  = 1;    // 1, 2, 3, or 4
};

// Packed uniform buffer uploaded to GPU each frame.
// This struct is mirrored exactly in Metal as `constant FrameUniforms& frame`.
struct alignas(16) FrameUniforms {
    float time;
    float delta_time;
    uint32_t frame_index;
    uint32_t flags;            // bit 0: show_grid

    float resolution[2];
    float inv_resolution[2];

    float mouse[2];
    float mouse_click[2];

    // Camera
    float camera_pos[3];
    float _pad1;
    float camera_fwd[3];
    float _pad2;
    float camera_up[3];
    float _pad3;
    float camera_right[3];
    float camera_fov;

    // View-projection matrices (column-major, 4x4)
    float view_proj[16];
    float prev_view_proj[16];
    float inv_view_proj[16];

    // Jitter for TAA
    float jitter[2];
    float _pad4[2];

    // Shader params: each param occupies one float4 regardless of actual size.
    float params[32][4];
    uint32_t param_count;
    uint32_t _pad5[3];
};

// Texture descriptor for backend allocation
struct TextureDesc {
    uint32_t width;
    uint32_t height;
    enum Format { RGBA16Float, RGBA32Float, R32Float, R32Uint } format;
    bool read_write;
    std::string name;
};
