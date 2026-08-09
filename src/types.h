#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <cstring>

// Parsed from shader comment headers
struct ShaderParam {
    enum Type { Float, Int, Float2, Float3, Float4, Enum };

    std::string name;
    Type type;
    bool is_color = false;       // true for color3/color4 — renders as ColorEdit
    float min_val[4]     = {};
    float max_val[4]     = {};
    float default_val[4] = {};
    float current_val[4] = {};
    int component_count  = 1;    // 1, 2, 3, or 4
    std::vector<std::string> labels;  // Enum only — combo entries; current_val[0] is the index
};

// Packed uniform buffer uploaded to GPU each frame.
// This struct is mirrored exactly in Metal as `constant FrameUniforms& frame`.
struct alignas(16) FrameUniforms {
    float time;
    float delta_time;
    uint32_t frame_index;
    uint32_t flags;            // bit 0: show_grid, bit 1: camera moved this frame

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

    // Per-iteration IFS rotation (column-major 3x3, one float4 per column),
    // built once per frame from params[3] — the DE runs thousands of times
    // per pixel and rebuilding it per call was ~40% of DE cost.
    float rot_mtx[3][4];

    // Shader params: each param occupies one float4 regardless of actual size.
    float params[32][4];           // raymarch.metal params
    float recon_params[12][4];     // reconstruct.metal params
    float pt_params[16][4];        // pathtrace.metal params
    uint32_t param_count;
    uint32_t recon_param_count;
    uint32_t pt_param_count;
    uint32_t accum_frames;         // frames accumulated in history (0 = invalidated this frame)

    // Post params live after accum_frames, outside the params_changed memcmp
    // span — grading tweaks must not reset accumulation.
    float post_params[8][4];       // present.metal params
    uint32_t post_param_count;
};

// Texture descriptor for backend allocation
struct TextureDesc {
    uint32_t width;
    uint32_t height;
    enum Format { RGBA16Float, RGBA32Float, R32Float, R32Uint } format;
    bool read_write;
    std::string name;
};
