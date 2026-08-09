#include <metal_stdlib>
using namespace metal;

// ---- Uniform struct (must mirror FrameUniforms in types.h exactly) ----
struct FrameUniforms {
    float  time;
    float  delta_time;
    uint   frame_index;
    uint   flags;              // bit 0: show_grid, bit 1: camera moved this frame

    float2 resolution;
    float2 inv_resolution;

    float2 mouse;
    float2 mouse_click;

    packed_float3 camera_pos;    float _pad1;
    packed_float3 camera_fwd;    float _pad2;
    packed_float3 camera_up;     float _pad3;
    packed_float3 camera_right;
    float  camera_fov;

    float4x4 view_proj;
    float4x4 prev_view_proj;
    float4x4 inv_view_proj;

    float2 jitter;
    float2 _pad4;

    float4 rot_mtx[3];          // per-iteration IFS rotation, columns (from params[3])

    float4 params[32];          // raymarch.metal params
    float4 recon_params[12];    // reconstruct.metal params
    float4 pt_params[16];       // pathtrace.metal params
    uint   param_count;
    uint   recon_param_count;
    uint   pt_param_count;
    uint   accum_frames;        // frames accumulated in history (0 = invalidated this frame)
};

// ---- SDF Primitives ----
float sd_sphere(float3 p, float r) {
    return length(p) - r;
}

float sd_box(float3 p, float3 b) {
    float3 d = abs(p) - b;
    return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, 0.0));
}

// ---- Rotation ----
// Rotation matrix around an arbitrary axis (angle in radians)
float3x3 rot_axis(float3 axis, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    float t = 1.0 - c;
    float3 a = normalize(axis);
    return float3x3(
        float3(t*a.x*a.x + c,     t*a.x*a.y - s*a.z, t*a.x*a.z + s*a.y),
        float3(t*a.x*a.y + s*a.z, t*a.y*a.y + c,      t*a.y*a.z - s*a.x),
        float3(t*a.x*a.z - s*a.y, t*a.y*a.z + s*a.x, t*a.z*a.z + c)
    );
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

// ---- Scene DE (shared by raymarch + pathtrace) ----
// Param slots (declared in raymarch.metal): [0] scale, [1] surface_color,
// [2] offset, [3] rotation, [4] marchRatio, [5] fold_limit, [6] min_radius,
// [7] box_dims, [8] levels, [9] lod_factor
float2 tglad(float3 z0, constant FrameUniforms& frame)
{
    int levels = int(frame.params[8].x);
    float s = frame.params[0].x;
    float fold_limit = frame.params[5].x;
    float min_r2 = frame.params[6].x;
    float3 box_dims = frame.params[7].xyz;

    float4 scale = float4(-s, -s, -s, s), p0 = frame.params[2].xyzz;
    float4 z = float4(z0, 1.0);
    float orbit = 0;

    // Per-iteration rotation, prebuilt on the CPU from params[3]
    float3x3 rot = float3x3(frame.rot_mtx[0].xyz, frame.rot_mtx[1].xyz, frame.rot_mtx[2].xyz);

    for (int n = 0; n < levels; n++)
    {
        float3 start = z.xyz;

        z.xyz = clamp(z.xyz, -fold_limit, fold_limit) * 2.0 - z.xyz;
        z *= scale / clamp(dot(z.xyz, z.xyz), min_r2, 1.0);
        z.xyz = rot * z.xyz;
        z += p0;
        orbit += length(start - z.xyz);
    }

    float dS = length(max(abs(z.xyz) - box_dims, 0)) / z.w;
    return float2(dS, orbit);
}

float2 DE(float3 p, constant FrameUniforms& frame) {
    return tglad(p, frame);
}

float3 calc_normal(float3 p, float t, constant FrameUniforms& frame) {
    // Tetrahedron gradient: 4 DE taps instead of 6
    float eps = 0.0001 * t;
    const float2 k = float2(1, -1);
    float3 g = k.xyy * DE(p + k.xyy * eps, frame).x +
               k.yyx * DE(p + k.yyx * eps, frame).x +
               k.yxy * DE(p + k.yxy * eps, frame).x +
               k.xxx * DE(p + k.xxx * eps, frame).x;
    // Fold seams and eps underflow at tiny t can zero the gradient;
    // normalize(0) is NaN and would poison the accumulator.
    float len = length(g);
    return (len > 0.0) ? (g / len) : float3(0, 1, 0);
}

// ---- Sphere trace ----
constant float TRACE_MAX_DIST = 100.0;

struct TraceResult {
    float t;      // >= TRACE_MAX_DIST on miss
    int steps;
};

// min_dist is relative to t (screen-space-ish). Secondary rays pass fewer
// steps and a looser epsilon — diffuse GI doesn't need primary precision.
//
// LOD termination (GPU Unchained [c]): the acceptance radius grows with the
// step count, so expensive rays (grazing angles, deep IFS interiors) hit a
// slightly inflated surface early instead of burning the full step budget.
// Caps worst-case cost per ray — trades a touch of blur for stable frame
// time. lod_scale lets bounce rays use a coarser LOD than primaries.
//
// t_start seeds the march partway along the ray (see seed_primary_t); 0
// preserves the classic march from the origin.
TraceResult trace(float3 origin, float3 dir, constant FrameUniforms& frame,
                  int max_steps, float min_dist, float lod_scale, float t_start) {
    float lod = frame.params[9].x * lod_scale;
    float t = t_start;
    int i = 0;
    for (; i < max_steps; i++) {
        float3 p = origin + dir * t;
        // Escape bailout: the IFS is bounded, so outside r = 20 heading
        // outward is a guaranteed miss — skip marching out to TRACE_MAX_DIST.
        if (dot(p, p) > 400.0 && dot(p, dir) > 0.0) {
            t = TRACE_MAX_DIST + 1.0;
            break;
        }
        float2 d = DE(p, frame);
        if (d.x < (min_dist + float(i) * lod) * t) break;
        if (t > TRACE_MAX_DIST) break;
        t += d.x * frame.params[4].x;
    }

    TraceResult r;
    r.t = t;
    r.steps = i;
    return r;
}

// ---- Sparse-stride fill ordering (GPU Unchained [e], "in the noise") ----
// A raster visit order refreshes every k x k block in the same sweep, which
// reads as a coherent scrolling front. Permuting the cycle with a multiplier
// coprime to k*k dithers the order (diagonal for 2x2, corners-then-edges for
// 3x3, bit-reversed-ish for 4x4) so holes are spatially scattered. Pathtrace
// maps cycle time -> cell offset; reconstruct inverts it to age each texel.
uint stride_fill_mult(uint stride) {
    return stride == 2u ? 3u : (stride == 3u ? 2u : (stride == 4u ? 7u : 1u));
}
uint stride_fill_inv(uint stride) {  // modular inverse of stride_fill_mult
    return stride == 2u ? 3u : (stride == 3u ? 5u : (stride == 4u ? 7u : 1u));
}
uint2 stride_cell_pos(uint t, uint stride) {
    uint c = (t * stride_fill_mult(stride)) % (stride * stride);
    return uint2(c % stride, c / stride);
}
uint stride_cell_time(uint2 texel, uint stride) {
    uint c = (texel.y % stride) * stride + (texel.x % stride);
    return (c * stride_fill_inv(stride)) % (stride * stride);
}

// ---- Primary-ray depth seeding (GPU Unchained [e], "bread crumbs") ----
// When the camera is static, last frame's reconstructed depth at this pixel
// is this ray's depth — start the march just short of it instead of at the
// origin. Jitter moves the ray up to a full-res pixel per frame, so take the
// conservative min over a 3x3 full-res neighborhood, then back off 10%.
// "If data isn't ready, just compute it": any guard failing means a plain
// march from zero, never a wait.
float seed_primary_t(texture2d<float, access::read> prev_depth, uint2 half_pixel,
                     constant FrameUniforms& frame) {
    // accum_frames == 0 -> history (and the depth ping-pong) was invalidated
    // this frame; moving flag -> prev depth is from a different camera pose.
    if (frame.accum_frames == 0u || (frame.flags & 2u) != 0u) return 0.0;

    int2 full_res = int2(prev_depth.get_width(), prev_depth.get_height());
    int2 center   = int2(half_pixel) * 2 + 1;
    float t_min = TRACE_MAX_DIST;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int2 q = clamp(center + int2(dx, dy), int2(0), full_res - 1);
            t_min = min(t_min, prev_depth.read(uint2(q)).r);
        }
    }
    // All-sky neighborhood: no surface to seed toward, and sky rays exit
    // cheaply via the escape bailout anyway.
    if (t_min >= TRACE_MAX_DIST) return 0.0;
    return max(t_min * 0.9, 0.0);
}

// ---- Subpixel sample offset, packed into color alpha ----
// 5 bits per axis over [-0.5, 0.5) half-texel units; integers <= 1023 are
// exact in half floats. Reconstruct reads back where each sample really is,
// which stays correct when sparse PT leaves stale texels behind.
float pack_offset(float2 delta) {
    uint2 q = uint2(clamp((delta + 0.5) * 32.0, 0.0, 31.0));
    return float(q.x + q.y * 32u);
}

float2 unpack_offset(float packed) {
    uint p = uint(packed);
    return (float2(uint2(p % 32u, p / 32u)) + 0.5) * (1.0 / 32.0) - 0.5;
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
