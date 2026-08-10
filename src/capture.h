#pragma once
#include <cstdint>
#include <cstddef>
#include <string>

// ---- Offline frame capture ----
//
// `tex_output` is RGBA16Float and present.metal already writes sRGB-encoded
// values into it (present.metal:52-55), so encoding is a straight
// clamp-and-quantize — no transfer function is applied here.
//
// `src` points at the shared MTLBuffer a texture->buffer blit filled, so rows
// are `bytes_per_row` apart (256-byte aligned), not tightly packed.
bool save_rgba16f_png(const void* src, uint32_t w, uint32_t h, size_t bytes_per_row,
                      bool sixteen_bit, const std::string& path);

// Metal requires texture->buffer blit rows to be 256-byte aligned.
inline size_t aligned_row_bytes(uint32_t w, uint32_t bytes_per_pixel) {
    size_t row = (size_t)w * bytes_per_pixel;
    return (row + 255) & ~(size_t)255;
}
