#pragma once
#include <cstdint>
#include <string>

// ---- Offline render job ----
//
// Animation time is pinned per output frame and only advances once that frame
// has accumulated `samples` traces. That inverts the interactive problem: the
// accumulation reset that fights animated params (main.cpp params_changed)
// becomes exactly what we want at output-frame boundaries, and is suppressed
// in between.
//
// Motion blur then costs almost nothing: within one output frame each sample
// is taken at a different time inside the shutter interval, and reconstruct's
// existing unclamped running mean (reconstruct.metal:112-131) integrates them.
// That is ground truth blur, not a post-process approximation.
struct RenderJob {
    bool        active = false;
    int         frame = 0;
    int         frame_first = 0;
    int         frame_last = 0;
    uint32_t    samples = 256;      // accumulation samples per output frame
    float       shutter = 0.5f;     // fraction of a frame interval; 0 = no blur
    float       scale = 1.0f;       // render resolution multiplier
    bool        png16 = false;
    std::string dir;

    // Runtime
    uint32_t sample = 0;
    bool     needs_reset = true;
    bool     capture_pending = false;   // blit issued this frame, read after end_frame
    int      failures = 0;
    double   started_at = 0.0;
    double   frame_started_at = 0.0;
    double   avg_frame_secs = 0.0;

    int  total_frames() const { return frame_last - frame_first + 1; }
    int  done_frames()  const { return frame - frame_first; }
    float progress() const {
        int n = total_frames();
        if (n <= 0) return 1.0f;
        return ((float)done_frames() + (float)sample / (float)(samples ? samples : 1)) / (float)n;
    }

    // Centered, stratified shutter offset in frames for the current sample.
    // Stratified rather than random because the sample count is known up
    // front, so we can cover the interval exactly instead of clumping.
    float shutter_offset() const {
        if (shutter <= 0.0f || samples == 0) return 0.0f;
        return shutter * (((float)sample + 0.5f) / (float)samples - 0.5f);
    }
};
