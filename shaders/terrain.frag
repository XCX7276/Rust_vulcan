#version 460
// Terrain fragment shader.
// TEMPORARY: flat block colors + simple sun shading + distance fog; textures, shadows and AO come later.

layout(push_constant) uniform Push {
    mat4 view_proj;
    ivec4 camera_block;
    vec4 camera_frac;
    vec4 sun_dir;
    vec4 fog_color;
} pc;

layout(location = 0) in vec3 v_color;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in float v_distance;

layout(location = 0) out vec4 out_color;

void main() {
    vec3 n = normalize(v_normal);
    float sun = max(dot(n, normalize(pc.sun_dir.xyz)), 0.0);
    // Faces never go fully dark: ambient keeps shaded sides readable.
    float sky = 0.55 + 0.25 * n.y;
    vec3 lit = v_color * (sky + 0.6 * sun);

    float fog_end = pc.camera_frac.w;
    float fog = clamp((v_distance - fog_end * 0.55) / (fog_end * 0.45), 0.0, 1.0);
    out_color = vec4(mix(lit, pc.fog_color.rgb, fog * fog), 1.0);
}
