// Param slot order is load-bearing: common.metal reads these by index, so
// never reorder or insert — append only. The `@if` clause is display-only,
// gating which sliders the GUI shows for the selected fractal (a hidden
// param keeps its value and is still uploaded, just unread by that DE).
//
// @param scale float -4.0 4.0 3.0                                            @if fractal=tglad,mandelbox,menger,simple
// @param offset float3 -5 -5 -5 5 5 5 0.9 0.6 0.3                            @if fractal=tglad,menger,pkleinian,simple
// @param rotation float3 -3.14 -3.14 -3.14 3.14 3.14 3.14 0.0 0.0 0.0        @if fractal=tglad,menger,simple
// @param marchRatio float 0.1 1.0 0.9
// @param fold_limit float 0.1 2.0 1.0                                        @if fractal=tglad,mandelbox
// @param min_radius float 0.05 1.0 0.25                                      @if fractal=tglad,mandelbox
// @param box_dims float3 0 0 0 50 50 50 1.0 50.0 1.4                         @if fractal=tglad,simple
// @param levels int 1 20 6
// @param fractal enum tglad mandelbulb mandelbox menger pkleinian simple
// @param power float 2 16 8                                                  @if fractal=mandelbulb
// @param csize float3 0 0 0 2 2 2 0.9 0.9 0.9                                @if fractal=pkleinian
// @param ksize float 0.1 2.0 1.0                                             @if fractal=pkleinian
// @param color_amp color3 0 0 0 1 1 1 0.5 0.5 0.5
// @param color_freq float3 0 0 0 4 4 4 1.0 1.0 1.0
// @param color_phase float3 0 0 0 1 1 1 0.0 0.33 0.67
// @param orbit_scale float 0.01 4.0 0.3
// @param lod_factor float 0.0 0.0002 0.00002
// @param mat_freq float 0 8 2.0
// @param mat_phase float 0 1 0.0
// @param rough_range float2 0 0 1 1 0.15 0.7                                 @if renderer=pathtrace
// @param metal_range float2 0 0 1 1 0.0 0.0                                  @if renderer=pathtrace
// @param emission_gain float 0 20 0.0
// @param emission_width float 0 1 0.15

// No depth seeding here (unlike pathtrace): this kernel's shading is the
// step count itself, so seeded marches would brighten the image whenever
// the camera stops. texture(2) is bound by the shared dispatch but unused.
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

    TraceResult tr = trace(ray.origin, ray.direction, frame, 128, 0.0001, 1.0, 0.0);

    float3 color = float3(0.0);
    float depth = TRACE_MAX_DIST;

    if (tr.t < TRACE_MAX_DIST) {
        depth = tr.t;
        Material mat = orbit_material(tr.orbit, frame);
        color = mat.albedo * pow(1.0 - float(tr.steps) / 128.0, 2.0) + mat.emission;
    }

    out_color.write(float4(color, pack_offset(frame.jitter)), gid);
    out_depth.write(float4(depth, 0, 0, 0), gid);
}
