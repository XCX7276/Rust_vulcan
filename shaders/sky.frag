#version 460
// Procedural sky: horizon -> zenith gradient, glow and disc of the sun, disc of the moon.

// Per-frame data (FrameData in frame_data.rs, std140; keep both in sync).
layout(std140, set = 0, binding = 0) uniform Frame {
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

layout(location = 0) in vec2 v_ndc;
layout(location = 0) out vec4 out_color;

// Disc pixels across the radius: the sun and the moon are 16 x 16 pixel-art circles.
const float DISC_PIXELS = 8.0;

// Pixel coordinates of the view ray on a disc of angular radius `radius` around `body`:
// returns the pixel cell (integer, -8..7) and whether that pixel belongs to the disc.
bool disc_pixel(vec3 dir, vec3 body, float radius, out vec2 cell) {
    cell = vec2(0.0);
    float facing = dot(dir, body);
    if (facing <= 0.0) {
        return false;
    }
    // Disc axes: "right" stays horizontal, so pixel rows are parallel to the horizon.
    vec3 helper = abs(body.y) > 0.999 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
    vec3 right = normalize(cross(helper, body));
    vec3 up = cross(body, right);
    vec2 p = vec2(dot(dir, right), dot(dir, up)) / facing / tan(radius);
    cell = floor(p * DISC_PIXELS);
    vec2 center = (cell + 0.5) / DISC_PIXELS;
    return dot(center, center) <= 1.0;
}

float hash(vec2 cell) {
    return fract(sin(dot(cell, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    // View ray: un-project a point of this pixel (camera at the origin).
    vec4 far_point = frame.inv_view_proj * vec4(v_ndc, 0.5, 1.0);
    vec3 dir = normalize(far_point.xyz / far_point.w);
    vec3 color = sky_color(dir);

    float day = frame.sun_dir.w;
    vec2 cell;
    // Sun: bright core, warmer rim pixels; fades in and out at the horizon.
    if (disc_pixel(dir, frame.sun_dir.xyz, frame.sun_glow.w, cell)) {
        vec2 c = (cell + 0.5) / DISC_PIXELS;
        bool rim = dot(c, c) > 0.62;
        vec3 sun = rim ? vec3(4.2, 3.0, 1.4) : vec3(6.0, 5.4, 4.4);
        color = mix(color, sun, smoothstep(-0.03, 0.02, frame.sun_dir.y));
    }
    // Moon: pale disc with a few darker "crater" pixels; visible at night.
    if (disc_pixel(dir, frame.moon_dir.xyz, frame.moon_dir.w, cell)) {
        bool crater = hash(cell) < 0.16;
        vec3 moon = crater ? vec3(0.42, 0.44, 0.50) : vec3(0.80, 0.82, 0.88);
        color = mix(color, moon, (1.0 - day) * smoothstep(-0.03, 0.02, frame.moon_dir.y));
    }
    out_color = vec4(color, 1.0);
}
