#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <cstring>

// Parsed from shader comment headers
struct ShaderParam {
    enum Type { Float, Int, Float2, Float3, Float4, Enum };

    // Visibility gate parsed from a trailing `@if <name>=<v1>,<v2>` clause.
    // `name` is either another param in the same shader (an enum, matched
    // against its current label) or a host-supplied context key such as
    // `renderer`. `!=` inverts. Multiple clauses are ANDed.
    struct Condition {
        std::string name;
        std::vector<std::string> values;
        bool negate = false;
    };

    std::string name;
    Type type;
    bool is_color = false;       // true for color3/color4 — renders as ColorEdit
    // min/max are the LIVE slider range, which the range popup may widen past
    // what the shader declared. Everything that draws a slider or a curve axis
    // reads these, so an override propagates everywhere for free.
    float min_val[4]     = {};
    float max_val[4]     = {};
    // The range as declared in the shader — the "Reset range" target, and the
    // reference that makes "is this overridden?" a derived question.
    float decl_min[4]    = {};
    float decl_max[4]    = {};
    float default_val[4] = {};
    float current_val[4] = {};
    int component_count  = 1;    // 1, 2, 3, or 4
    bool logarithmic     = false;  // ImGuiSliderFlags_Logarithmic
    std::vector<std::string> labels;  // Enum only — combo entries; current_val[0] is the index
    std::vector<Condition> conditions;  // empty = always shown
    // Panel section from a trailing `@group <name>`. Display-only: the GUI
    // draws sections in first-appearance order, so related params can sit
    // together while declaration (slot) order stays append-only.
    std::string group;                  // "" = ungrouped
};

// An enum's range is derived from its label list, and ColorEdit ignores range
// entirely — neither is meaningfully overridable, so both are locked out of
// the range UI, of serialization, and of the hot-reload merge.
inline bool param_range_editable(const ShaderParam& p) {
    return p.type != ShaderParam::Enum && !p.is_color;
}

// Derived, never stored: a stored bool could only go stale against min/max.
inline bool has_range_override(const ShaderParam& p) {
    return param_range_editable(p) &&
           (p.min_val[0] != p.decl_min[0] || p.max_val[0] != p.decl_max[0] || p.logarithmic);
}

// Packed uniform buffer uploaded to GPU each frame.
// This struct is mirrored exactly in Metal as `constant FrameUniforms& frame`.
struct alignas(16) FrameUniforms {
    float time;
    float delta_time;
    uint32_t frame_index;
    // bit 0: show_grid
    // bit 1: camera moved this frame (clamped TAA path + no depth seeding)
    // bit 2: offline render with an open shutter (no depth seeding, but the
    //        unclamped accumulator stays on — it's what integrates the blur)
    uint32_t flags;

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
