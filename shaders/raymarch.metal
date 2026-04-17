// @param radius float 0.1 5.0 1.0
// @param surface_color color3 0 0 0 1 1 1 0.9 0.6 0.3

float DE(float3 p, constant FrameUniforms& frame) {
    float radius = frame.params[0].x;
    return sd_sphere(p, radius);
}

// Grid + axes on the XZ plane (y=0)
float3 render_grid(float3 hit_pos, float t) {
    float2 xz = hit_pos.xz;

    // Anti-aliased grid lines — scale line width with distance
    float2 grid = abs(fract(xz - 0.5) - 0.5);
    float line_width = max(0.01, 0.002 * t);  // thicken with distance
    float2 line = smoothstep(float2(0.0), float2(line_width), grid);
    float grid_mask = 1.0 - min(line.x, line.y);

    // Base grid color (dark gray lines on darker bg)
    float3 color = mix(float3(0.02), float3(0.12), grid_mask);

    // Axis lines — highlight X (red) and Z (blue) axes
    float axis_width = max(0.03, 0.005 * t);
    float x_axis = smoothstep(axis_width, 0.0, abs(xz.y));  // Z=0 line → X axis
    float z_axis = smoothstep(axis_width, 0.0, abs(xz.x));  // X=0 line → Z axis

    color = mix(color, float3(0.8, 0.1, 0.1), x_axis);  // red for X
    color = mix(color, float3(0.1, 0.1, 0.8), z_axis);  // blue for Z

    // Y axis: small green dot cluster at origin
    float origin_dist = length(xz);
    float y_marker = smoothstep(axis_width * 2.0, 0.0, origin_dist);
    color = mix(color, float3(0.1, 0.8, 0.1), y_marker);  // green at origin

    return color;
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

    // Raymarch the SDF
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

    // SDF hit
    float3 color = float3(0.0);
    float depth = max_dist;

    if (t < max_dist) {
        depth = t;
        float3 hit_pos = ray.origin + ray.direction * t;

        float eps = 0.0001 * t;
        float3 n = normalize(float3(
            DE(hit_pos + float3(eps, 0, 0), frame) - DE(hit_pos - float3(eps, 0, 0), frame),
            DE(hit_pos + float3(0, eps, 0), frame) - DE(hit_pos - float3(0, eps, 0), frame),
            DE(hit_pos + float3(0, 0, eps), frame) - DE(hit_pos - float3(0, 0, eps), frame)
        ));

        float3 light_dir = normalize(float3(0.5, 1.0, -0.3));
        float ndl = max(dot(n, light_dir), 0.0);
        float ambient = 0.15;

        float3 surface_col = frame.params[1].xyz;
        color = surface_col * (ndl + ambient);
    }

    // Grid plane — ray-plane intersection at y=0
    bool show_grid = (frame.flags & 1u) != 0;
    if (show_grid && ray.direction.y != 0.0) {
        float t_plane = -ray.origin.y / ray.direction.y;
        if (t_plane > 0.0 && t_plane < depth) {
            float3 hit = ray.origin + ray.direction * t_plane;
            color = render_grid(hit, t_plane);
            depth = t_plane;
        }
    }

    out_color.write(float4(color, 1.0), gid);
    out_depth.write(float4(depth, 0, 0, 0), gid);
}
