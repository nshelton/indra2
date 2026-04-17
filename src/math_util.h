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

} // namespace v3
