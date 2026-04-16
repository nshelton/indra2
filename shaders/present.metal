// Simple hash for film grain noise
float hash12(float2 p) {
    float3 p3 = fract(float3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

kernel void present_kernel(
    texture2d<float, access::read>    in_color    [[texture(0)]],
    texture2d<float, access::read>    in_depth    [[texture(1)]],
    texture2d<float, access::read>    history     [[texture(2)]],
    texture2d<float, access::write>   output      [[texture(3)]],
    constant FrameUniforms&           frame       [[buffer(0)]],
    uint2                             gid         [[thread_position_in_grid]]
) {
    uint2 full_res = uint2(output.get_width(), output.get_height());
    if (gid.x >= full_res.x || gid.y >= full_res.y) return;

    // Bilinear fetch from half-res source
    float2 half_uv = float2(gid) * 0.5;
    uint2 half_coord = uint2(clamp(half_uv, float2(0), float2(in_color.get_width() - 1, in_color.get_height() - 1)));

    float4 color = in_color.read(half_coord);

    // Film grain noise
    float noise_amount = 0.02;
    float noise = hash12(float2(gid) + frame.time * 137.0) * 2.0 - 1.0;
    color.rgb += noise * noise_amount;

    // Clamp output
    color = max(color, float4(0.0));
    color.a = 1.0;

    output.write(color, gid);
}
