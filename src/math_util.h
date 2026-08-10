#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>

// Halton low-discrepancy sequence
inline float halton(uint32_t index, uint32_t base) {
    float f = 1.0f, r = 0.0f;
    uint32_t i = index;
    while (i > 0) {
        f /= (float)base;
        r += f * (float)(i % base);
        i /= base;
    }
    return r;
}

// Minimal column-major 4x4 matrix utilities (matching Metal layout)
namespace mat4 {

inline void identity(float* out) {
    std::memset(out, 0, 16 * sizeof(float));
    out[0] = out[5] = out[10] = out[15] = 1.0f;
}

inline void multiply(const float* a, const float* b, float* out) {
    float tmp[16];
    for (int c = 0; c < 4; c++) {
        for (int r = 0; r < 4; r++) {
            float sum = 0.0f;
            for (int k = 0; k < 4; k++) {
                sum += a[k * 4 + r] * b[c * 4 + k];
            }
            tmp[c * 4 + r] = sum;
        }
    }
    std::memcpy(out, tmp, 16 * sizeof(float));
}

inline void perspective(float fov_y, float aspect, float near_z, float far_z, float* out) {
    std::memset(out, 0, 16 * sizeof(float));
    float t = std::tan(fov_y * 0.5f);
    out[0]  = 1.0f / (aspect * t);
    out[5]  = 1.0f / t;
    out[10] = far_z / (near_z - far_z);
    out[11] = -1.0f;
    out[14] = (far_z * near_z) / (near_z - far_z);
}

inline void look_at(const float* eye, const float* center, const float* up, float* out) {
    float fx = center[0] - eye[0];
    float fy = center[1] - eye[1];
    float fz = center[2] - eye[2];
    float fl = std::sqrt(fx*fx + fy*fy + fz*fz);
    fx /= fl; fy /= fl; fz /= fl;

    // right = normalize(cross(f, up))
    float rx = fy * up[2] - fz * up[1];
    float ry = fz * up[0] - fx * up[2];
    float rz = fx * up[1] - fy * up[0];
    float rl = std::sqrt(rx*rx + ry*ry + rz*rz);
    rx /= rl; ry /= rl; rz /= rl;

    // u = cross(r, f)
    float ux = ry * fz - rz * fy;
    float uy = rz * fx - rx * fz;
    float uz = rx * fy - ry * fx;

    // Column-major
    out[0] = rx;  out[4] = ry;  out[8]  = rz;  out[12] = -(rx*eye[0] + ry*eye[1] + rz*eye[2]);
    out[1] = ux;  out[5] = uy;  out[9]  = uz;  out[13] = -(ux*eye[0] + uy*eye[1] + uz*eye[2]);
    out[2] = -fx; out[6] = -fy; out[10] = -fz; out[14] =  (fx*eye[0] + fy*eye[1] + fz*eye[2]);
    out[3] = 0;   out[7] = 0;   out[11] = 0;   out[15] = 1.0f;
}

inline bool invert(const float* m, float* out) {
    float inv[16];
    inv[0]  =  m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15] + m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10];
    inv[4]  = -m[4]*m[10]*m[15] + m[4]*m[11]*m[14] + m[8]*m[6]*m[15] - m[8]*m[7]*m[14] - m[12]*m[6]*m[11] + m[12]*m[7]*m[10];
    inv[8]  =  m[4]*m[9]*m[15]  - m[4]*m[11]*m[13] - m[8]*m[5]*m[15] + m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[9];
    inv[12] = -m[4]*m[9]*m[14]  + m[4]*m[10]*m[13] + m[8]*m[5]*m[14] - m[8]*m[6]*m[13] - m[12]*m[5]*m[10] + m[12]*m[6]*m[9];
    inv[1]  = -m[1]*m[10]*m[15] + m[1]*m[11]*m[14] + m[9]*m[2]*m[15] - m[9]*m[3]*m[14] - m[13]*m[2]*m[11] + m[13]*m[3]*m[10];
    inv[5]  =  m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15] + m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10];
    inv[9]  = -m[0]*m[9]*m[15]  + m[0]*m[11]*m[13] + m[8]*m[1]*m[15] - m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[9];
    inv[13] =  m[0]*m[9]*m[14]  - m[0]*m[10]*m[13] - m[8]*m[1]*m[14] + m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[9];
    inv[2]  =  m[1]*m[6]*m[15]  - m[1]*m[7]*m[14]  - m[5]*m[2]*m[15] + m[5]*m[3]*m[14] + m[13]*m[2]*m[7]  - m[13]*m[3]*m[6];
    inv[6]  = -m[0]*m[6]*m[15]  + m[0]*m[7]*m[14]  + m[4]*m[2]*m[15] - m[4]*m[3]*m[14] - m[12]*m[2]*m[7]  + m[12]*m[3]*m[6];
    inv[10] =  m[0]*m[5]*m[15]  - m[0]*m[7]*m[13]  - m[4]*m[1]*m[15] + m[4]*m[3]*m[13] + m[12]*m[1]*m[7]  - m[12]*m[3]*m[5];
    inv[14] = -m[0]*m[5]*m[14]  + m[0]*m[6]*m[13]  + m[4]*m[1]*m[14] - m[4]*m[2]*m[13] - m[12]*m[1]*m[6]  + m[12]*m[2]*m[5];
    inv[3]  = -m[1]*m[6]*m[11]  + m[1]*m[7]*m[10]  + m[5]*m[2]*m[11] - m[5]*m[3]*m[10] - m[9]*m[2]*m[7]   + m[9]*m[3]*m[6];
    inv[7]  =  m[0]*m[6]*m[11]  - m[0]*m[7]*m[10]  - m[4]*m[2]*m[11] + m[4]*m[3]*m[10] + m[8]*m[2]*m[7]   - m[8]*m[3]*m[6];
    inv[11] = -m[0]*m[5]*m[11]  + m[0]*m[7]*m[9]   + m[4]*m[1]*m[11] - m[4]*m[3]*m[9]  - m[8]*m[1]*m[7]   + m[8]*m[3]*m[5];
    inv[15] =  m[0]*m[5]*m[10]  - m[0]*m[6]*m[9]   - m[4]*m[1]*m[10] + m[4]*m[2]*m[9]  + m[8]*m[1]*m[6]   - m[8]*m[2]*m[5];

    float det = m[0]*inv[0] + m[1]*inv[4] + m[2]*inv[8] + m[3]*inv[12];
    if (std::abs(det) < 1e-12f) return false;

    float inv_det = 1.0f / det;
    for (int i = 0; i < 16; i++) out[i] = inv[i] * inv_det;
    return true;
}

} // namespace mat4

// Column-major 3x3, matching Metal float3x3 layout and rot_axis in common.metal
namespace mat3 {

inline void axis_rotation(const float* axis, float angle, float* out) {
    float c = std::cos(angle), s = std::sin(angle), t = 1.0f - c;
    float ax = axis[0], ay = axis[1], az = axis[2];
    out[0] = t*ax*ax + c;    out[1] = t*ax*ay - s*az; out[2] = t*ax*az + s*ay;  // column 0
    out[3] = t*ax*ay + s*az; out[4] = t*ay*ay + c;    out[5] = t*ay*az - s*ax;  // column 1
    out[6] = t*ax*az - s*ay; out[7] = t*ay*az + s*ax; out[8] = t*az*az + c;     // column 2
}

inline void multiply(const float* a, const float* b, float* out) {
    for (int j = 0; j < 3; j++)
        for (int i = 0; i < 3; i++)
            out[j*3+i] = a[0+i]*b[j*3+0] + a[3+i]*b[j*3+1] + a[6+i]*b[j*3+2];
}

} // namespace mat3

// Minimal quaternion for trackball rotation (w, x, y, z)
namespace quat {

inline void from_axis_angle(const float* axis, float angle, float* out) {
    float half = angle * 0.5f;
    float s = std::sin(half);
    out[0] = std::cos(half); // w
    out[1] = axis[0] * s;    // x
    out[2] = axis[1] * s;    // y
    out[3] = axis[2] * s;    // z
}

// Rotate a vector by a quaternion: q * v * q_conjugate
inline void rotate_vec3(const float* q, const float* v, float* out) {
    // q * (0, v) * conj(q)
    // Expand using quaternion multiplication
    float qw = q[0], qx = q[1], qy = q[2], qz = q[3];

    // t = 2 * cross(q.xyz, v)
    float tx = 2.0f * (qy * v[2] - qz * v[1]);
    float ty = 2.0f * (qz * v[0] - qx * v[2]);
    float tz = 2.0f * (qx * v[1] - qy * v[0]);

    // result = v + qw * t + cross(q.xyz, t)
    out[0] = v[0] + qw * tx + (qy * tz - qz * ty);
    out[1] = v[1] + qw * ty + (qz * tx - qx * tz);
    out[2] = v[2] + qw * tz + (qx * ty - qy * tx);
}

} // namespace quat

// Float3 helpers (inline, no types)
namespace v3 {

inline float dot(const float* a, const float* b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

inline float length(const float* v) {
    return std::sqrt(dot(v, v));
}

inline void normalize(const float* v, float* out) {
    float l = length(v);
    if (l > 1e-8f) { out[0] = v[0]/l; out[1] = v[1]/l; out[2] = v[2]/l; }
    else { out[0] = out[1] = out[2] = 0; }
}

inline void cross(const float* a, const float* b, float* out) {
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

inline void sub(const float* a, const float* b, float* out) {
    out[0] = a[0]-b[0]; out[1] = a[1]-b[1]; out[2] = a[2]-b[2];
}

inline void add(const float* a, const float* b, float* out) {
    out[0] = a[0]+b[0]; out[1] = a[1]+b[1]; out[2] = a[2]+b[2];
}

inline void scale(const float* v, float s, float* out) {
    out[0] = v[0]*s; out[1] = v[1]*s; out[2] = v[2]*s;
}

inline void mad(const float* a, const float* b, float s, float* out) {
    // out = a + b * s
    out[0] = a[0]+b[0]*s; out[1] = a[1]+b[1]*s; out[2] = a[2]+b[2]*s;
}

inline void lerp(const float* a, const float* b, float t, float* out) {
    out[0] = a[0] + (b[0]-a[0])*t; out[1] = a[1] + (b[1]-a[1])*t; out[2] = a[2] + (b[2]-a[2])*t;
}

// Spherical interpolation of two unit vectors. Near-parallel (and near-
// antiparallel, where the rotation plane is undefined) falls back to nlerp:
// sin(theta) goes to zero in the denominator, and over a tiny arc the chord
// and the arc are the same thing anyway.
inline void slerp(const float* a, const float* b, float t, float* out) {
    float d = dot(a, b);
    d = d < -1.0f ? -1.0f : (d > 1.0f ? 1.0f : d);
    if (d > 0.9995f || d < -0.9995f) {
        lerp(a, b, t, out);
        normalize(out, out);
        return;
    }
    float theta = std::acos(d);
    float s = std::sin(theta);
    float wa = std::sin((1.0f - t) * theta) / s;
    float wb = std::sin(t * theta) / s;
    out[0] = a[0]*wa + b[0]*wb;
    out[1] = a[1]*wa + b[1]*wb;
    out[2] = a[2]*wa + b[2]*wb;
    normalize(out, out);
}

} // namespace v3

// ---- Scalar interpolation (animation curves) ----

// Uniform (non-centripetal) Catmull-Rom through p1..p2, with p0/p3 as the
// neighbouring keys. Endpoint segments duplicate their outer key, which makes
// the curve start and end with zero-ish overshoot instead of flying off.
inline float catmull_rom(float p0, float p1, float p2, float p3, float t) {
    float t2 = t * t, t3 = t2 * t;
    return 0.5f * ((2.0f*p1) +
                   (-p0 + p2) * t +
                   (2.0f*p0 - 5.0f*p1 + 4.0f*p2 - p3) * t2 +
                   (-p0 + 3.0f*p1 - 3.0f*p2 + p3) * t3);
}

// Cubic Hermite. m0/m1 are tangents in value-units per second, dt is the
// segment length in seconds, t is normalized 0..1 across it.
inline float hermite(float v0, float v1, float m0, float m1, float dt, float t) {
    float t2 = t * t, t3 = t2 * t;
    float h00 =  2.0f*t3 - 3.0f*t2 + 1.0f;
    float h10 =       t3 - 2.0f*t2 + t;
    float h01 = -2.0f*t3 + 3.0f*t2;
    float h11 =       t3 -      t2;
    return h00*v0 + h10*(m0*dt) + h01*v1 + h11*(m1*dt);
}
