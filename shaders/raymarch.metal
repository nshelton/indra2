// @param radius float 0.1 5.0 1.0
// @param surface_color color3 0 0 0 1 1 1 0.9 0.6 0.3
// @param offset float3 -5 -5 -5 5 5 5 0.9 0.6 0.3
// @param rotation float3 -3.14 -3.14 -3.14 3.14 3.14 3.14 0.0 0.0 0.0

float hash12(float2 p) {
    float3 p3 = fract(float3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float2 tglad(float3 z0, constant FrameUniforms& frame)
{
     z0 = fmod(z0, 1.0);
    int _LEVELS = 6;

    float s = 3;
    float4 scale = float4(-s, -s, -s, s), p0 = frame.params[2].xyzz;
    float4 z = float4(z0, 1.0);
    float orbit = 0;

    // Per-iteration rotation (param index 3: x, y, z angles)
    float3 rot_angles = frame.params[3].xyz;
    float3x3 rot_x = rot_axis(float3(1, 0, 0), rot_angles.x);
    float3x3 rot_y = rot_axis(float3(0, 1, 0), rot_angles.y);
    float3x3 rot_z = rot_axis(float3(0, 0, 1), rot_angles.z);
    float3x3 rot = rot_x * rot_y * rot_z;

    for (int n = 0; n < _LEVELS; n++)
    {
        float3 start = z.xyz;

        z.xyz = clamp(z.xyz, -1, 1) * 2.0 - z.xyz;
        z *= scale / clamp(dot(z.xyz, z.xyz),  0.25, 1.0);
        z.xyz = rot * z.xyz;
        z += p0;
        orbit += length(start - z.xyz);
    }

    float dS = (length(max(abs(z.xyz) - float3(1.0, 50.0, 1.4), 0)) ) / z.w;
    return float2(dS, orbit);
}

float2 DE(float3 p, constant FrameUniforms& frame) {
    float radius = frame.params[0].x;
    return tglad(p, frame);
}

kernel void raymarch_kernel(
    texture2d<float, access::write>   out_color  [[texture(0)]],
    texture2d<float, access::write>   out_depth  [[texture(1)]],
    constant FrameUniforms&           frame      [[buffer(0)]],
    uint2                             gid        [[thread_position_in_grid]]
) {
    uint2 half_res = uint2(out_color.get_width(), out_color.get_height());
    if (gid.x >= half_res.x || gid.y >= half_res.y) return;

    float2 full_pixel = (float2(gid) + 0.5) * 2.0 + frame.jitter ;
    Ray ray = make_camera_ray(full_pixel, frame);

    float noise = hash12(full_pixel);
 
    // Raymarch the SDF
    float t = 0;
    float2 d = float2(0.0, 0.0);
    int max_steps = 128;
    float min_dist = 0.0001;
    float max_dist = 100.0;

    int iter = max_steps;

    for (int i = 0; i < max_steps; i++) {
        float3 p = ray.origin + ray.direction * t;
        d = DE(p, frame);
        if (d.x < min_dist * t) break;
        if (t > max_dist) break;
        t += d.x;
        iter--;
    }

    // SDF hit
    float3 color = float3(0.0);
    float depth = max_dist;

    if (t < max_dist) {
        depth = t;
        float3 hit_pos = ray.origin + ray.direction * t;

        float eps = 0.0001 * t;
        float3 n = normalize(float3(
            DE(hit_pos + float3(eps, 0, 0), frame).x - DE(hit_pos - float3(eps, 0, 0), frame).x,
            DE(hit_pos + float3(0, eps, 0), frame).x - DE(hit_pos - float3(0, eps, 0), frame).x,
            DE(hit_pos + float3(0, 0, eps), frame).x - DE(hit_pos - float3(0, 0, eps), frame).x
        ));

        float3 light_dir = normalize(float3(0.5, 1.0, 0.5));
        float ndl = max(dot(n, light_dir), 0.0);
    

        float3 surface_col = frame.params[1].xyz;
        color = surface_col;
    }

    color *= pow(float(iter) / float(max_steps), 2.0);

    out_color.write(float4(color, 1.0), gid);
    out_depth.write(float4(depth, 0, 0, 0), gid);
}
