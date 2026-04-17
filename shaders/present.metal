kernel void present_kernel(
    texture2d<float, access::read>   in_color [[texture(0)]],  // full-res, from reconstruct
    texture2d<float, access::write>  output   [[texture(1)]],  // full-res
    constant FrameUniforms&          frame    [[buffer(0)]],
    uint2                            gid      [[thread_position_in_grid]]
) {
    uint2 full_res = uint2(output.get_width(), output.get_height());
    if (gid.x >= full_res.x || gid.y >= full_res.y) return;

    float4 color = in_color.read(gid);

    // TODO: tonemap / exposure / vignette go here

    output.write(color, gid);
}
