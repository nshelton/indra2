// Placeholder for TAA reconstruction kernel.
// Will be implemented after scaffold is validated.

kernel void reconstruct_kernel(
    texture2d<float, access::read>    current_color  [[texture(0)]],
    texture2d<float, access::read>    current_depth  [[texture(1)]],
    texture2d<float, access::read>    history_in     [[texture(2)]],
    texture2d<float, access::write>   history_out    [[texture(3)]],
    texture2d<float, access::write>   output         [[texture(4)]],
    constant FrameUniforms&           frame          [[buffer(0)]],
    uint2                             gid            [[thread_position_in_grid]]
) {
    uint2 res = uint2(output.get_width(), output.get_height());
    if (gid.x >= res.x || gid.y >= res.y) return;

    // Passthrough for now
    float2 half_uv = float2(gid) * 0.5;
    uint2 half_coord = uint2(clamp(half_uv, float2(0), float2(current_color.get_width() - 1, current_color.get_height() - 1)));
    float4 color = current_color.read(half_coord);

    output.write(color, gid);
    history_out.write(color, gid);
}
