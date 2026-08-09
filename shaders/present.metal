// @param exposure_ev   float -6.0 6.0 0.0
// @param contrast      float 0.5 2.0 1.0
// @param tonemap       enum none reinhard aces
// @param distortion_k1 float -0.5 0.5 0.0
// @param ca_amount     float 0.0 0.02 0.0

// ACES filmic fit (Narkowicz 2015)
static float3 aces_tonemap(float3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

static float3 linear_to_srgb(float3 c) {
    return select(1.055 * pow(c, 1.0 / 2.4) - 0.055, c * 12.92, c <= 0.0031308);
}

kernel void present_kernel(
    texture2d<float, access::sample> in_color [[texture(0)]],  // full-res, from reconstruct
    texture2d<float, access::write>  output   [[texture(1)]],  // full-res
    constant FrameUniforms&          frame    [[buffer(0)]],
    uint2                            gid      [[thread_position_in_grid]]
) {
    uint2 full_res = uint2(output.get_width(), output.get_height());
    if (gid.x >= full_res.x || gid.y >= full_res.y) return;

    float exposure = exp2(frame.post_params[0].x);
    float contrast = frame.post_params[1].x;
    int   tonemap  = (int)frame.post_params[2].x;
    float k1       = frame.post_params[3].x;
    float ca       = frame.post_params[4].x;

    // Lens distortion + lateral CA: aspect-corrected centered coords, Brown
    // radial term, CA as per-channel magnification on the distorted coord.
    float2 res = float2(full_res);
    float aspect = res.x / res.y;
    float2 uv = (float2(gid) + 0.5) / res;
    float2 p = float2((uv.x - 0.5) * aspect, uv.y - 0.5);
    float2 pd = p * (1.0 + k1 * dot(p, p));

    constexpr sampler lin(filter::linear, address::clamp_to_edge, coord::normalized);
    float2 uv_r = float2(pd.x * (1.0 + ca) / aspect, pd.y * (1.0 + ca)) + 0.5;
    float2 uv_g = float2(pd.x / aspect,              pd.y)              + 0.5;
    float2 uv_b = float2(pd.x * (1.0 - ca) / aspect, pd.y * (1.0 - ca)) + 0.5;

    float3 c = float3(in_color.sample(lin, uv_r).r,
                      in_color.sample(lin, uv_g).g,
                      in_color.sample(lin, uv_b).b) * exposure;

    if      (tonemap == 1) c = c / (1.0 + c);   // Reinhard
    else if (tonemap == 2) c = aces_tonemap(c);

    c = linear_to_srgb(saturate(c));
    c = saturate((c - 0.5) * contrast + 0.5);

    output.write(float4(c, 1.0), gid);
}
