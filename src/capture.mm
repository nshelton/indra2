#import <CoreGraphics/CoreGraphics.h>
#import <ImageIO/ImageIO.h>

#include "capture.h"
#include <cstring>
#include <vector>

// IEEE 754 half -> float. Written out rather than leaning on __fp16 so the
// conversion is unambiguous about denormals and infinities.
static inline float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;

    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;                       // +/- zero
        } else {
            // Denormal: renormalize into a float exponent.
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; exp--; }
            mant &= 0x3FFu;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13);   // inf / NaN
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }

    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

static inline float saturate(float v) {
    // NaN-safe: the comparison ordering sends NaN to 0 rather than through.
    return v > 0.0f ? (v < 1.0f ? v : 1.0f) : 0.0f;
}

bool save_rgba16f_png(const void* src, uint32_t w, uint32_t h, size_t bytes_per_row,
                      bool sixteen_bit, const std::string& path) {
    if (!src || w == 0 || h == 0) return false;

    const size_t bpc = sixteen_bit ? 2 : 1;
    const size_t dst_row = (size_t)w * 4 * bpc;
    std::vector<uint8_t> out(dst_row * h);

    for (uint32_t y = 0; y < h; y++) {
        const uint16_t* s = (const uint16_t*)((const uint8_t*)src + (size_t)y * bytes_per_row);
        uint8_t* d = out.data() + (size_t)y * dst_row;
        if (sixteen_bit) {
            uint16_t* d16 = (uint16_t*)d;
            for (uint32_t x = 0; x < w * 4; x++) {
                d16[x] = (uint16_t)(saturate(half_to_float(s[x])) * 65535.0f + 0.5f);
            }
        } else {
            for (uint32_t x = 0; x < w * 4; x++) {
                d[x] = (uint8_t)(saturate(half_to_float(s[x])) * 255.0f + 0.5f);
            }
        }
    }

    CFDataRef data = CFDataCreate(nullptr, out.data(), (CFIndex)out.size());
    if (!data) return false;
    CGDataProviderRef provider = CGDataProviderCreateWithCFData(data);
    CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);

    // The alpha channel carries the accumulation weight, not coverage — skip
    // it. present.metal already applied the sRGB transfer function, so tagging
    // sRGB here is metadata only; CGImageCreate performs no conversion.
    CGBitmapInfo info = kCGImageAlphaNoneSkipLast;
    if (sixteen_bit) info |= kCGBitmapByteOrder16Little;

    CGImageRef img = CGImageCreate(w, h, bpc * 8, bpc * 8 * 4, dst_row, cs, info, provider,
                                   nullptr, false, kCGRenderingIntentDefault);

    bool ok = false;
    if (img) {
        CFURLRef url = CFURLCreateFromFileSystemRepresentation(
            nullptr, (const UInt8*)path.c_str(), (CFIndex)path.size(), false);
        if (url) {
            CGImageDestinationRef dest =
                CGImageDestinationCreateWithURL(url, CFSTR("public.png"), 1, nullptr);
            if (dest) {
                CGImageDestinationAddImage(dest, img, nullptr);
                ok = CGImageDestinationFinalize(dest);
                CFRelease(dest);
            }
            CFRelease(url);
        }
        CGImageRelease(img);
    }

    CGColorSpaceRelease(cs);
    CGDataProviderRelease(provider);
    CFRelease(data);
    return ok;
}
