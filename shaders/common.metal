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

    float4 rot_mtx[3];          // per-iteration IFS rotation, columns (from params[2])

    float4 params[32];          // raymarch.metal params
    float4 recon_params[12];    // reconstruct.metal params
    float4 pt_params[16];       // pathtrace.metal params
    uint   param_count;
    uint   recon_param_count;
    uint   pt_param_count;
    uint   accum_frames;        // frames accumulated in history (0 = invalidated this frame)

    float4 post_params[8];      // present.metal params
    uint   post_param_count;
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
// Param slots (declared in raymarch.metal): [0] scale, [1] offset,
// [2] rotation, [3] marchRatio, [4] fold_limit, [5] min_radius,
// [6] box_dims, [7] levels ([8]+ continue below; [16] lod_factor,
// [17]+ material, [23] fixed_radius)
//
// min_radius and fixed_radius are both the *squared* radii of the sphere
// fold, matching the classic mandelbox parameterization (0.25 / 1.0).
float2 tglad(float3 z0, constant FrameUniforms& frame)
{
    int levels = int(frame.params[7].x);
    float s = frame.params[0].x;
    float fold_limit = frame.params[4].x;
    float min_r2 = frame.params[5].x;
    // max() so the clamp below stays well-defined when the sliders cross —
    // Metal's clamp is undefined for minval > maxval.
    float fixed_r2 = max(frame.params[23].x, min_r2);
    float3 box_dims = frame.params[6].xyz;

    float4 scale = float4(-s, -s, -s, s), p0 = frame.params[1].xyzz;
    float4 z = float4(z0, 1.0);
    float orbit = 0;

    // Per-iteration rotation, prebuilt on the CPU from params[2]
    float3x3 rot = float3x3(frame.rot_mtx[0].xyz, frame.rot_mtx[1].xyz, frame.rot_mtx[2].xyz);

    for (int n = 0; n < levels; n++)
    {
        float3 start = z.xyz;

        z.xyz = clamp(z.xyz, -fold_limit, fold_limit) * 2.0 - z.xyz;
        z *= scale / clamp(dot(z.xyz, z.xyz), min_r2, fixed_r2);
        z.xyz = rot * z.xyz;
        z += p0;
        orbit += length(start - z.xyz);
    }

    float dS = length(max(abs(z.xyz) - box_dims, 0)) / z.w;
    return float2(dS, orbit);
}

// Param slots continued: [8] fractal, [9] power, [10] csize, [11] ksize
float2 mandelbulb(float3 pos, constant FrameUniforms& frame) {
    int levels = int(frame.params[7].x);
    float power = frame.params[9].x;

    float3 z = pos;
    float dr = 1.0;
    float r = 0.0;
    float orbit = 1e9;

    for (int i = 0; i < levels; i++) {
        r = length(z);
        if (r > 2.0 || r < 1e-9) break;
        orbit = min(orbit, r);

        float theta = acos(z.z / r) * power;
        float phi = atan2(z.y, z.x) * power;
        dr = pow(r, power - 1.0) * power * dr + 1.0;

        z = pow(r, power) * float3(sin(theta) * cos(phi),
                                   sin(theta) * sin(phi),
                                   cos(theta)) + pos;
    }

    return float2(0.5 * log(r) * r / dr, orbit);
}

float2 mandelbox(float3 pos, constant FrameUniforms& frame) {
    int levels = int(frame.params[7].x);
    float s = frame.params[0].x;
    float fold_limit = frame.params[4].x;
    float min_r2 = frame.params[5].x;
    float fixed_r2 = frame.params[23].x;
    float3 off = frame.params[1].xyz;
    float3x3 rot = float3x3(frame.rot_mtx[0].xyz, frame.rot_mtx[1].xyz, frame.rot_mtx[2].xyz);

    float3 z = pos;
    float dr = 1.0;
    float orbit = 1e9;

    for (int i = 0; i < levels; i++) {
        z = box_fold(z, fold_limit);
        sphere_fold(z, dr, min_r2, fixed_r2);
        z = rot * z;                  // twist (identity when rotation = 0)
        // off shifts the julia constant: c = pos + off, zero is the classic
        // box. Rotation is an isometry and off is constant per iteration, so
        // dr's running derivative needs no change for either.
        z = s * z + pos + off;
        dr = dr * abs(s) + 1.0;
        orbit = min(orbit, length(z));
    }

    return float2(length(z) / abs(dr), orbit);
}

float2 menger(float3 pos, constant FrameUniforms& frame) {
    int levels = int(frame.params[7].x);
    float s = frame.params[0].x;
    float3 off = frame.params[1].xyz;
    float3x3 rot = float3x3(frame.rot_mtx[0].xyz, frame.rot_mtx[1].xyz, frame.rot_mtx[2].xyz);

    float3 z = pos;
    float dr = 1.0;
    float orbit = 1e9;

    for (int i = 0; i < levels; i++) {
        z = abs(z);
        if (z.x < z.y) z.xy = z.yx;
        if (z.x < z.z) z.xz = z.zx;
        if (z.y < z.z) z.yz = z.zy;
        z = rot * z;
        z = z * s - off * (s - 1.0);
        if (z.z < -0.5 * off.z * (s - 1.0)) z.z += off.z * (s - 1.0);
        dr *= abs(s);
        orbit = min(orbit, length(z));
    }

    return float2(sd_box(z, float3(1.0)) / dr, orbit);
}

float2 pkleinian(float3 pos, constant FrameUniforms& frame) {
    int levels = int(frame.params[7].x);
    float3 csize = frame.params[10].xyz;
    float ksize = frame.params[11].x;
    float3 c = frame.params[1].xyz;

    float3 p = pos;
    float defactor = 1.0;
    float orbit = 1e9;

    for (int i = 0; i < levels; i++) {
        p = 2.0 * clamp(p, -csize, csize) - p;
        float r2 = dot(p, p);
        float k = max(ksize / r2, 1.0);
        p *= k;
        defactor *= k;
        p += c;
        orbit = min(orbit, length(p));
    }

    return float2(0.5 * abs(p.z) / defactor, orbit);
}

// Minimal kaleidoscopic IFS — the skeleton every DE fractal here shares:
// FOLD space so many regions map onto one, SCALE about a fixed point, repeat.
// The result is self-similar because each iteration re-runs the same map on
// an ever-smaller copy of space. dr tracks the total stretch so the distance
// to the base shape (a unit box) can be converted back to world units.
float2 simple(float3 pos, constant FrameUniforms& frame) {
    int levels = int(frame.params[7].x);
    float s = frame.params[0].x;
    float3 off = frame.params[1].xyz;
    float3 box_dims = frame.params[6].xyz;
    float3x3 rot = float3x3(frame.rot_mtx[0].xyz, frame.rot_mtx[1].xyz, frame.rot_mtx[2].xyz);

    float3 z = pos;
    float dr = 1.0;
    float orbit = 1e9;

    for (int i = 0; i < levels; i++) {
        z = abs(z);                   // fold: mirror all 8 octants into one
        z = rot * z;                  // twist (identity when rotation = 0)
        z = z * s - off * (s - 1.0);  // scale about the point `off`
        dr *= abs(s);
        orbit = min(orbit, length(z.xz));  // line trap: distance to the y axis
    }

    return float2(sd_box(z, box_dims) / dr, orbit);
}

float2 DE(float3 p, constant FrameUniforms& frame) {
    switch (int(frame.params[8].x)) {
        case 1:  return mandelbulb(p, frame);
        case 2:  return mandelbox(p, frame);
        case 3:  return menger(p, frame);
        case 4:  return pkleinian(p, frame);
        case 5:  return simple(p, frame);
        default: return tglad(p, frame);
    }
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
    float orbit;  // orbit trap from the last DE eval — the hit point's when t is a hit
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
    float lod = frame.params[16].x * lod_scale;  // slot 16: lod_factor
    float t = t_start;
    float orbit = 0;
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
        orbit = d.y;
        if (d.x < (min_dist + float(i) * lod) * t) break;
        if (t > TRACE_MAX_DIST) break;
        t += d.x * frame.params[3].x;
    }

    TraceResult r;
    r.t = t;
    r.steps = i;
    r.orbit = orbit;
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
// Per-block phase offset: without it every block traces the same cell each
// frame and the fresh pixels form a screen-wide lattice. Hashing the block
// coords desynchronizes the cycles so fresh pixels are spatially scattered.
uint stride_block_phase(uint2 block, uint stride) {
    uint h = block.x * 1664525u + block.y * 1013904223u;
    h = (h ^ (h >> 16u)) * 0x45d9f3bu;
    h = h ^ (h >> 16u);
    return h % (stride * stride);
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
    // this frame; moving flag (bit 1) -> prev depth is from a different camera
    // pose; offline shutter (bit 2) -> the camera moves between accumulation
    // samples, and the surface can close in by more than the 10% back-off.
    if (frame.accum_frames == 0u || (frame.flags & 6u) != 0u) return 0.0;

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

// ---- Orbit-driven material ----
// One color source and one modulation wave, both functions of
// t = orbit * orbit_scale[15]:
//   palette = saturate(color_amp[12] * (1 + cos(2pi(color_freq[13]*t + color_phase[14]))))
//     — fully procedural, peaks at 2*amp, troughs at black. It is the albedo,
//     the metal tint, and the emission color.
//   wave w = cos bands at mat_freq[17]/mat_phase[18] — roughness[19] and
//     metallic[20] lerp across their float2 ranges by w; emission ignites at
//     wave crests (width [22], gain [21]) in the local palette color.
struct Material {
    float3 albedo;
    float  metallic;
    float  roughness;
    float3 emission;
};

Material orbit_material(float orbit, constant FrameUniforms& frame) {
    float t = orbit * frame.params[15].x;
    float3 palette = saturate(frame.params[12].xyz *
        (1.0 + cos(6.2831853 * (frame.params[13].xyz * t + frame.params[14].xyz))));

    float w = 0.5 + 0.5 * cos(6.2831853 * (frame.params[17].x * t + frame.params[18].x));

    Material m;
    m.albedo    = palette;
    m.roughness = mix(frame.params[19].x, frame.params[19].y, w);
    m.metallic  = mix(frame.params[20].x, frame.params[20].y, w);
    // max() keeps smoothstep's edges apart at width 0 — degenerate edges are
    // undefined and 0 * NaN would still poison the accumulator.
    float glow  = smoothstep(1.0 - max(frame.params[22].x, 1e-3), 1.0, w);
    m.emission  = palette * (frame.params[21].x * glow);
    return m;
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
