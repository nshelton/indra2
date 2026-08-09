// @param scale float 1.0 4.0 3.0
// @param surface_color color3 0 0 0 1 1 1 0.9 0.6 0.3
// @param offset float3 -5 -5 -5 5 5 5 0.9 0.6 0.3
// @param rotation float3 -3.14 -3.14 -3.14 3.14 3.14 3.14 0.0 0.0 0.0
// @param marchRatio float 0.1 1.0 0.9
// @param fold_limit float 0.1 2.0 1.0
// @param min_radius float 0.05 1.0 0.25
// @param box_dims float3 0 0 0 3 60 3 1.0 50.0 1.4
// @param levels int 1 10 6

kernel void raymarch_kernel(
    texture2d<float, access::write>   out_color  [[texture(0)]],
    texture2d<float, access::write>   out_depth  [[texture(1)]],
    constant FrameUniforms&           frame      [[buffer(0)]],
    uint2                             gid        [[thread_position_in_grid]]
) {
    uint2 half_res = uint2(out_color.get_width(), out_color.get_height());
    if (gid.x >= half_res.x || gid.y >= half_res.y) return;

    // jitter * 2 spans the full 2x2 quad; reconstruct's jitter-aware weights
    // re-center it, and accumulation resolves genuine full-res detail.
    float2 full_pixel = (float2(gid) + 0.5) * 2.0 + frame.jitter * 2.0;
    Ray ray = make_camera_ray(full_pixel, frame);

    TraceResult tr = trace(ray.origin, ray.direction, frame, 128, 0.0001);

    float3 color = float3(0.0);
    float depth = TRACE_MAX_DIST;

    if (tr.t < TRACE_MAX_DIST) {
        depth = tr.t;
        color = frame.params[1].xyz;
    }

    color *= pow(1.0 - float(tr.steps) / 128.0, 2.0);

    out_color.write(float4(color, pack_offset(frame.jitter)), gid);
    out_depth.write(float4(depth, 0, 0, 0), gid);
}
