#include <metal_stdlib>
using namespace metal;

// ---- Uniform struct (must mirror FrameUniforms in types.h exactly) ----
struct FrameUniforms {
    float  time;
    float  delta_time;
    uint   frame_index;
    uint   _pad0;

    float2 resolution;
    float2 inv_resolution;

    float2 mouse;
    float2 mouse_click;

    float3 camera_pos;    float _pad1;
    float3 camera_fwd;    float _pad2;
    float3 camera_up;     float _pad3;
    float3 camera_right;
    float  camera_fov;

    float4x4 view_proj;
    float4x4 prev_view_proj;
    float4x4 inv_view_proj;

    float2 jitter;
    float2 _pad4;

    float4 params[32];
    uint   param_count;
    uint3  _pad5;
};

// ---- SDF Primitives ----
float sd_sphere(float3 p, float r) {
    return length(p) - r;
}

float sd_box(float3 p, float3 b) {
    float3 d = abs(p) - b;
    return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, 0.0));
}

// ---- Fractal Utilities ----
float3 box_fold(float3 p, float fold_limit) {
    return clamp(p, -fold_limit, fold_limit) * 2.0 - p;
}

void sphere_fold(thread float3& p, thread float& dr, float min_r2, float fixed_r2) {
    float r2 = dot(p, p);
    if (r2 < min_r2) {
        float ratio = fixed_r2 / min_r2;
        p *= ratio;
        dr *= ratio;
    } else if (r2 < fixed_r2) {
        float ratio = fixed_r2 / r2;
        p *= ratio;
        dr *= ratio;
    }
}

// ---- Ray generation ----
struct Ray {
    float3 origin;
    float3 direction;
};

Ray make_camera_ray(float2 pixel_pos, constant FrameUniforms& frame) {
    float2 ndc = (pixel_pos * frame.inv_resolution) * 2.0 - 1.0;
    ndc.y = -ndc.y;

    float aspect = frame.resolution.x * frame.inv_resolution.y;
    float half_fov = tan(frame.camera_fov * 0.5);

    float3 dir = normalize(
        frame.camera_fwd +
        frame.camera_right * ndc.x * half_fov * aspect +
        frame.camera_up * ndc.y * half_fov
    );

    Ray r;
    r.origin = frame.camera_pos;
    r.direction = dir;
    return r;
}
