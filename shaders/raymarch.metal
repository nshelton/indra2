// @param radius float 0.1 5.0 1.0
// @param surface_color color3 0 0 0 1 1 1 0.9 0.6 0.3

float DE(float3 p, constant FrameUniforms& frame) {
    float radius = frame.params[0].x;
    return sd_sphere(p, radius);
}

kernel void raymarch_kernel(
    texture2d<float, access::write>   out_color  [[texture(0)]],
    texture2d<float, access::write>   out_depth  [[texture(1)]],
    constant FrameUniforms&           frame      [[buffer(0)]],
    uint2                             gid        [[thread_position_in_grid]]
) {
    uint2 half_res = uint2(out_color.get_width(), out_color.get_height());
    if (gid.x >= half_res.x || gid.y >= half_res.y) return;

    float2 full_pixel = (float2(gid) + 0.5) * 2.0 + frame.jitter;
    Ray ray = make_camera_ray(full_pixel, frame);

    // Raymarch
    float t = 0.0;
    float d = 0.0;
    int max_steps = 128;
    float min_dist = 0.0001;
    float max_dist = 100.0;

    for (int i = 0; i < max_steps; i++) {
        float3 p = ray.origin + ray.direction * t;
        d = DE(p, frame);
        if (d < min_dist * t) break;
        if (t > max_dist) break;
        t += d;
    }

    // Output
    float3 color = float3(0.0);
    float depth = max_dist;

    if (t < max_dist) {
        depth = t;
        float3 hit_pos = ray.origin + ray.direction * t;

        // Normal via central differences
        float eps = 0.0001 * t;
        float3 n = normalize(float3(
            DE(hit_pos + float3(eps, 0, 0), frame) - DE(hit_pos - float3(eps, 0, 0), frame),
            DE(hit_pos + float3(0, eps, 0), frame) - DE(hit_pos - float3(0, eps, 0), frame),
            DE(hit_pos + float3(0, 0, eps), frame) - DE(hit_pos - float3(0, 0, eps), frame)
        ));

        // Hemisphere lighting
        float3 light_dir = normalize(float3(0.5, 1.0, -0.3));
        float ndl = max(dot(n, light_dir), 0.0);
        float ambient = 0.15;

        float3 surface_col = frame.params[1].xyz;
        color = surface_col * (ndl + ambient);
    }

    out_color.write(float4(color, 1.0), gid);
    out_depth.write(float4(depth, 0, 0, 0), gid);
}
