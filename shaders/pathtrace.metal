// @param max_bounces int 1 8 4
// @param sky_horizon color3 0 0 0 1 1 1 1.0 0.9 0.8
// @param sky_zenith color3 0 0 0 1 1 1 0.2 0.4 0.8
// @param sky_intensity float 0.0 10.0 1.0
// @param sample_stride int 1 4 1

uint pcg(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float rand01(thread uint& seed) {
    seed = pcg(seed);
    return float(seed) * (1.0 / 4294967296.0);
}

float3 cosine_hemisphere(float3 n, thread uint& seed) {
    float r1 = rand01(seed);
    float r2 = rand01(seed);
    float a = 6.2831853 * r1;
    float r = sqrt(r2);
    float3 t = normalize(cross(n, abs(n.y) < 0.99 ? float3(0, 1, 0) : float3(1, 0, 0)));
    float3 b = cross(n, t);
    return normalize(t * (r * cos(a)) + b * (r * sin(a)) + n * sqrt(max(1.0 - r2, 0.0)));
}

float3 sky(float3 dir, constant FrameUniforms& frame) {
    float h = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    return mix(frame.pt_params[1].xyz, frame.pt_params[2].xyz, h) * frame.pt_params[3].x;
}

kernel void pathtrace_kernel(
    texture2d<float, access::write>   out_color  [[texture(0)]],
    texture2d<float, access::write>   out_depth  [[texture(1)]],
    texture2d<float, access::read>    prev_depth [[texture(2)]],  // full-res, last frame's reconstructed depth
    constant FrameUniforms&           frame      [[buffer(0)]],
    uint2                             gid        [[thread_position_in_grid]]
) {
    uint2 half_res = uint2(out_color.get_width(), out_color.get_height());

    // Sparse sampling: stride k traces 1 of k*k pixels per frame, cycling
    // through the k x k block positions in a dithered fill order (see
    // stride_cell_pos) so every texel refreshes each k*k frames without a
    // visible raster sweep. main.cpp shrinks the dispatch grid to match.
    uint stride = clamp(uint(frame.pt_params[4].x), 1u, 4u);
    uint idx = frame.frame_index % (stride * stride);
    uint2 pixel = gid * stride + stride_cell_pos(idx, stride);
    if (pixel.x >= half_res.x || pixel.y >= half_res.y) return;

    uint seed = (pixel.x * 1973u + pixel.y * 9277u + frame.frame_index * 26699u) | 1u;

    // Frame-uniform jitter over the full 2x2 quad, matching raymarch —
    // each sample's offset is packed into color alpha so reconstruct can
    // weight it by its true position even after it goes stale.
    float2 full_pixel = (float2(pixel) + 0.5) * 2.0 + frame.jitter * 2.0;
    Ray ray = make_camera_ray(full_pixel, frame);

    float3 radiance = float3(0.0);
    float3 throughput = float3(1.0);
    float depth = TRACE_MAX_DIST;

    int max_bounces = int(frame.pt_params[0].x);
    float3 albedo = frame.params[1].xyz;
    float3 ro = ray.origin;
    float3 rd = ray.direction;

    // Bread crumbs: static camera lets primaries start just short of last
    // frame's reconstructed depth instead of marching from the origin.
    float t_seed = seed_primary_t(prev_depth, pixel, frame);

    for (int bounce = 0; bounce < max_bounces; bounce++) {
        // Bounce rays carry low-frequency diffuse GI — march them with a
        // smaller step budget, looser epsilon, and coarser LOD than primaries.
        TraceResult tr = (bounce == 0) ? trace(ro, rd, frame, 128, 0.0001, 1.0, t_seed)
                                       : trace(ro, rd, frame, 64, 0.0006, 4.0, 0.0);

        if (tr.t >= TRACE_MAX_DIST) {
            radiance += throughput * sky(rd, frame);
            break;
        }

        if (bounce == 0) depth = tr.t;

        float3 p = ro + rd * tr.t;
        float3 n = calc_normal(p, tr.t, frame);

        throughput *= albedo;

        // Russian roulette: kill dim paths unbiasedly instead of marching them
        if (bounce >= 2) {
            float p_survive = clamp(max(throughput.x, max(throughput.y, throughput.z)), 0.05, 1.0);
            if (rand01(seed) >= p_survive) break;
            throughput /= p_survive;
        }

        rd = cosine_hemisphere(n, seed);
        ro = p + n * (0.0003 * tr.t);
    }

    // NaN/negative guard via min/max, which drop NaN operands (arithmetic
    // doesn't) — one bad sample would poison the unclamped accumulator forever.
    radiance = min(max(radiance, float3(0.0)), float3(1e4));

    out_color.write(float4(radiance, pack_offset(frame.jitter)), pixel);
    out_depth.write(float4(depth, 0, 0, 0), pixel);
}
