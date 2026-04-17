// @param filter_sigma       float 0.1 3.0 1.0
// @param filter_depth_sigma float 0.001 0.5 0.05
// @param edge_aware         float 0.0 1.0 1.0
// @param taa_alpha          float 0.0 1.0 0.1
// @param taa_clamp_scale    float 0.1 3.0 1.0

kernel void reconstruct_kernel(
    texture2d<float, access::read>    current_color [[texture(0)]],  // half-res
    texture2d<float, access::read>    current_depth [[texture(1)]],  // half-res
    texture2d<float, access::read>    history_in    [[texture(2)]],  // full-res, unused in spatial-only
    texture2d<float, access::write>   history_out           [[texture(3)]],  // full-res
    texture2d<float, access::write>   reconstructed_depth   [[texture(4)]],  // full-res — dominant-tap depth
    constant FrameUniforms&           frame                 [[buffer(0)]],
    uint2                             gid           [[thread_position_in_grid]]
) {
    uint2 full_res = uint2(history_out.get_width(), history_out.get_height());
    if (gid.x >= full_res.x || gid.y >= full_res.y) return;

    float sigma           = frame.recon_params[0].x;
    float depth_sigma     = frame.recon_params[1].x;
    float edge_aware      = frame.recon_params[2].x;
    float taa_alpha       = frame.recon_params[3].x;
    float taa_clamp_scale = frame.recon_params[4].x;

    int2  half_res  = int2(current_color.get_width(), current_color.get_height());
    float2 half_coord = (float2(gid) + 0.5) * 0.5 - 0.5;
    int2   base       = int2(floor(half_coord));

    int2  ref_tap = clamp(int2(round(half_coord)), int2(0), half_res - 1);
    float z_ref   = current_depth.read(uint2(ref_tap)).r;

    float4 sum  = float4(0.0);
    float  wsum = 0.0;
    float  best_w    = -1.0;
    float  dom_depth = z_ref;  // fallback if loop never updates (shouldn't happen)
    float4 ngb_min   = float4(1e30);
    float4 ngb_max   = float4(-1e30);

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int2  tap    = clamp(base + int2(dx, dy), int2(0), half_res - 1);
            float2 off   = float2(tap) - half_coord;
            float  w_s   = exp(-0.5 * dot(off, off) / (sigma * sigma));

            float  z     = current_depth.read(uint2(tap)).r;
            float  z_rel = (z_ref > 1e-4) ? (abs(z - z_ref) / (z_ref * depth_sigma)) : 0.0;
            float  w_d   = mix(1.0, exp(-z_rel), edge_aware);

            float  w     = w_s * w_d;
            if (w > best_w) {
                best_w    = w;
                dom_depth = z;
            }
            float4 tap_color = current_color.read(uint2(tap));
            ngb_min          = min(ngb_min, tap_color);
            ngb_max          = max(ngb_max, tap_color);
            sum  += tap_color * w;
            wsum += w;
        }
    }

    float4 resolved_current = sum / max(wsum, 1e-6);

    // --- TAA: reproject + clamp + blend ---

    float4 history_clamped = resolved_current;
    float  alpha           = taa_alpha;

    // Edge cases that force TAA off for this pixel
    const float max_dist_threshold = 99.0;  // matches raymarch's max_dist = 100.0
    bool skip_taa = (frame.frame_index == 0u) || (dom_depth >= max_dist_threshold);

    if (!skip_taa) {
        // Build world-space hit point using the same camera basis as raymarch
        float2 ndc = (float2(gid) + 0.5) * frame.inv_resolution * 2.0 - 1.0;
        ndc.y = -ndc.y;
        float  aspect   = frame.resolution.x * frame.inv_resolution.y;
        float  half_fov = tan(frame.camera_fov * 0.5);
        float3 ray_dir  = normalize(
            frame.camera_fwd +
            frame.camera_right * ndc.x * half_fov * aspect +
            frame.camera_up    * ndc.y * half_fov
        );
        float3 world_pos = frame.camera_pos + ray_dir * dom_depth;

        float4 prev_clip = frame.prev_view_proj * float4(world_pos, 1.0);

        if (prev_clip.w <= 0.0) {
            alpha = 1.0;  // behind prev camera
        } else {
            float2 prev_ndc = prev_clip.xy / prev_clip.w;
            float2 prev_uv  = prev_ndc * 0.5 + 0.5;
            prev_uv.y       = 1.0 - prev_uv.y;  // match top-left pixel convention

            if (any(prev_uv < 0.0) || any(prev_uv > 1.0)) {
                alpha = 1.0;  // off-screen in prev frame
            } else {
                // Manual bilinear fetch from history_in
                float2 prev_pix = prev_uv * float2(history_in.get_width(), history_in.get_height());
                int2   i0       = int2(floor(prev_pix - 0.5));
                float2 f        = prev_pix - 0.5 - float2(i0);
                int2   hw       = int2(history_in.get_width(), history_in.get_height());
                int2   c00      = clamp(i0 + int2(0, 0), int2(0), hw - 1);
                int2   c10      = clamp(i0 + int2(1, 0), int2(0), hw - 1);
                int2   c01      = clamp(i0 + int2(0, 1), int2(0), hw - 1);
                int2   c11      = clamp(i0 + int2(1, 1), int2(0), hw - 1);
                float4 h00      = history_in.read(uint2(c00));
                float4 h10      = history_in.read(uint2(c10));
                float4 h01      = history_in.read(uint2(c01));
                float4 h11      = history_in.read(uint2(c11));
                float4 h0       = mix(h00, h10, f.x);
                float4 h1       = mix(h01, h11, f.x);
                float4 history  = mix(h0, h1, f.y);

                // AABB clamp with padding
                float4 ngb_center = 0.5 * (ngb_min + ngb_max);
                float4 ngb_half   = 0.5 * (ngb_max - ngb_min) * taa_clamp_scale;
                float4 ngb_lo     = ngb_center - ngb_half;
                float4 ngb_hi     = ngb_center + ngb_half;
                history_clamped   = clamp(history, ngb_lo, ngb_hi);
            }
        }
    } else {
        alpha = 1.0;
    }

    float4 final = mix(history_clamped, resolved_current, alpha);

    history_out.write(final, gid);
    reconstructed_depth.write(float4(dom_depth, 0, 0, 0), gid);
}
