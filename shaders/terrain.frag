#version 460
// Terrain fragment shader: sun/moon light with cascaded shadows, sky ambient with AO, height fog.
// TEMPORARY: flat block colors instead of textures.

// Per-frame data (FrameData in frame_data.rs, std140; keep both in sync).
layout(std140, set = 1, binding = 0) uniform Frame {
    mat4 view_proj;      // camera rotation + projection (camera at origin)
    mat4 inv_view_proj;  // inverse of view_proj (view rays for the sky)
    mat4 cascades[4];    // camera-relative position -> shadow map clip space, per cascade
    vec4 cascade_split;  // far distance of each cascade (blocks)
    vec4 cascade_texel;  // world size of one shadow texel in each cascade
    ivec4 camera_block;  // camera integer block position relative to the render origin
    vec4 camera_frac;    // xyz: fractional part of the camera position; w: view distance
    vec4 light_dir;      // xyz: direction to the light (sun or moon); w: 1 = shadows on
    vec4 light_color;    // rgb: light color * intensity; w: shadow distance
    vec4 ambient_sky;    // rgb: ambient from above; w: AO strength (0..1)
    vec4 ambient_ground; // rgb: ambient from below
    vec4 sky_zenith;     // rgb
    vec4 sky_horizon;    // rgb
    vec4 sun_dir;        // xyz: direction to the sun; w: day factor (0 night .. 1 day)
    vec4 moon_dir;       // xyz: direction to the moon
    vec4 sun_glow;       // rgb: glow around the sun near the horizon
    vec4 fog;            // x: density at base height, y: falloff per block, z: camera world Y, w: base height
    vec4 viewport;       // xy: viewport size in pixels
} frame;

layout(set = 1, binding = 1) uniform sampler2DArrayShadow shadow_map;

// Sky color along a view direction (also used as the fog color, so terrain fades into the sky).
vec3 sky_color(vec3 dir) {
    float up = max(dir.y, 0.0);
    vec3 color = mix(frame.sky_horizon.rgb, frame.sky_zenith.rgb, sqrt(up));
    // Below the horizon: slightly darker haze instead of a hard edge.
    color *= 1.0 - 0.35 * clamp(-dir.y * 4.0, 0.0, 1.0);
    float toward_sun = max(dot(dir, frame.sun_dir.xyz), 0.0);
    color += frame.sun_glow.rgb * pow(toward_sun, 6.0) * (1.0 - 0.7 * up);
    return color;
}

layout(location = 0) in vec3 v_color;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec3 v_position;
layout(location = 3) in float v_light;

layout(location = 0) out vec4 out_color;

// 1 = lit, 0 = in shadow.
float shadow(vec3 position, vec3 normal, float dist) {
    float shadow_end = frame.light_color.w;
    if (frame.light_dir.w < 0.5 || dist >= shadow_end) {
        return 1.0;
    }
    int cascade = 0;
    while (cascade < 3 && dist > frame.cascade_split[cascade]) {
        cascade++;
    }
    // Pixel shadows: look the shadow up at the centre of the block "pixel" (1/16 block, the texture
    // pixel of a block) on the world grid, so shadow edges are steps of block pixels running
    // parallel to the block edges. The same pixel size in every cascade: coarser pixels far away
    // made thin shadows flicker in and out when the camera moved across cascade borders.
    const float cell = 1.0 / 16.0;
    // camera_block is an integer, so camera_frac + position is on the same grid as the world.
    vec3 grid = position + frame.camera_frac.xyz;
    vec3 snapped = (floor(grid / cell) + 0.5) * cell - frame.camera_frac.xyz;
    // Snap only along the face; keep the face plane, plus a hair outwards.
    vec3 p = mix(snapped, position, abs(normal)) + normal * 0.02;
    vec4 light_clip = frame.cascades[cascade] * vec4(p, 1.0);
    vec2 uv = light_clip.xy * 0.5 + 0.5;
    float depth = light_clip.z;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0))) || depth > 1.0) {
        return 1.0;
    }
    float lit = texture(shadow_map, vec4(uv, float(cascade), depth));
    // Fade out towards the end of the shadow distance.
    return mix(lit, 1.0, smoothstep(shadow_end * 0.85, shadow_end, dist));
}

void main() {
    vec3 n = normalize(v_normal);
    float dist = length(v_position);
    vec3 view_dir = v_position / max(dist, 1.0e-4);

    // Direct light (sun by day, moon by night).
    float ndl = max(dot(n, frame.light_dir.xyz), 0.0);
    float lit = ndl > 0.0 ? shadow(v_position, n, dist) : 0.0;
    // Ambient: sky from above, ground bounce from below; AO only darkens the ambient part much.
    float ao = mix(1.0, v_light, frame.ambient_sky.w);
    vec3 ambient = mix(frame.ambient_ground.rgb, frame.ambient_sky.rgb, n.y * 0.5 + 0.5);
    vec3 light = frame.light_color.rgb * ndl * lit * mix(0.75, 1.0, ao) + ambient * mix(0.35, 1.0, ao);
    vec3 color = v_color * light;

    // Height fog (denser low down) + full fade at the end of the view distance.
    float density = frame.fog.x * exp(-frame.fog.y * (frame.fog.z - frame.fog.w));
    float rise = frame.fog.y * v_position.y;
    float path = abs(rise) > 1.0e-3 ? (1.0 - exp(-rise)) / rise : 1.0;
    float haze = 1.0 - exp(-density * dist * path);
    float view_distance = frame.camera_frac.w;
    float fog = max(haze, smoothstep(view_distance * 0.7, view_distance, dist));
    vec3 fog_color = sky_color(normalize(vec3(view_dir.x, max(view_dir.y, 0.0), view_dir.z)));
    out_color = vec4(mix(color, fog_color, fog), 1.0);
}
