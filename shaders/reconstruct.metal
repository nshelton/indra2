// @param filter_sigma       float 0.1 3.0 1.0
// @param filter_depth_sigma float 0.001 0.5 0.05
// @param edge_aware         float 0.0 1.0 1.0

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

    float sigma       = frame.recon_params[0].x;
    float depth_sigma = frame.recon_params[1].x;
    float edge_aware  = frame.recon_params[2].x;

    int2  half_res  = int2(current_color.get_width(), current_color.get_height());
    float2 half_coord = (float2(gid) + 0.5) * 0.5 - 0.5;
    int2   base       = int2(floor(half_coord));

    int2  ref_tap = clamp(int2(round(half_coord)), int2(0), half_res - 1);
    float z_ref   = current_depth.read(uint2(ref_tap)).r;

    float4 sum  = float4(0.0);
    float  wsum = 0.0;
    float  best_w    = -1.0;
    float  dom_depth = z_ref;  // fallback if loop never updates (shouldn't happen)

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
            sum  += current_color.read(uint2(tap)) * w;
            wsum += w;
        }
    }

    float4 resolved = sum / max(wsum, 1e-6);

    // TAA seam — no-op today. When temporal accumulation lands:
    //   float2 prev_uv = reproject(gid, frame);
    //   float4 history = history_in.read(clamp_uv(prev_uv));
    //   resolved = mix(history, resolved, alpha);

    history_out.write(resolved, gid);
    reconstructed_depth.write(float4(dom_depth, 0, 0, 0), gid);
}
