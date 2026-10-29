#version 460
// Terrain vertex shader (vertex pulling).
// There is no vertex buffer: each quad is 2 uints in the shared geometry pool, and 6 vertices
// (two triangles) are generated from it. gl_VertexIndex / 6 = quad, gl_VertexIndex % 6 = corner.
// gl_InstanceIndex = chunk slot (firstInstance of the indirect command written by cull.comp).

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

layout(std430, set = 0, binding = 0) readonly buffer Quads { uint quads[]; };
struct ChunkRecord {
    ivec4 origin;  // xyz: chunk corner relative to the render origin, w: 0 = free, else 1 + LOD level
    uvec4 mesh;    // x: first quad in the pool, y/z/w: quad counts per face (used by cull.comp)
};
layout(std430, set = 0, binding = 1) readonly buffer Chunks { ChunkRecord chunks[]; };
layout(std430, set = 0, binding = 2) readonly buffer Colors { vec4 block_color[]; };

const ivec3 NORMALS[6] = ivec3[](
    ivec3(1, 0, 0), ivec3(-1, 0, 0),
    ivec3(0, 1, 0), ivec3(0, -1, 0),
    ivec3(0, 0, 1), ivec3(0, 0, -1)
);
// Triangle orders: A and B wind the quad in opposite directions (chosen per face so that every
// face is counter-clockwise seen from outside); the "flip" variants use the other diagonal.
const ivec2 ORDER_A[6] = ivec2[](ivec2(0, 0), ivec2(1, 0), ivec2(1, 1), ivec2(0, 0), ivec2(1, 1), ivec2(0, 1));
const ivec2 ORDER_A_FLIP[6] = ivec2[](ivec2(0, 0), ivec2(1, 0), ivec2(0, 1), ivec2(1, 0), ivec2(1, 1), ivec2(0, 1));
const ivec2 ORDER_B[6] = ivec2[](ivec2(0, 0), ivec2(1, 1), ivec2(1, 0), ivec2(0, 0), ivec2(0, 1), ivec2(1, 1));
const ivec2 ORDER_B_FLIP[6] = ivec2[](ivec2(0, 0), ivec2(0, 1), ivec2(1, 0), ivec2(1, 0), ivec2(0, 1), ivec2(1, 1));

// AO "lightness" 0..3 of the quad corner at (s, t); corners are stored in the order
// (0,0), (1,0), (1,1), (0,1), 2 bits each.
uint corner_light(uint ao, ivec2 st) {
    uint index = st.y == 0 ? uint(st.x) : 3u - uint(st.x);
    return (ao >> (index * 2u)) & 3u;
}

struct Corner {
    vec3 position;   // relative to the camera
    vec3 normal;
    uint block;
    float light;     // AO lightness 0..1
    ivec3 e1;
    ivec3 e2;
    ivec2 st;
    uint face;
    int scale;
    // Depth priority among coplanar overlapping quads (see terrain.vert): 0 merged, 1-2 AO quads.
    int priority;
};

// Builds one vertex of a quad from the geometry pool (vertex pulling).
Corner pull_corner() {
    uint quad = uint(gl_VertexIndex) / 6u;
    uint corner_index = uint(gl_VertexIndex) % 6u;
    uint a = quads[quad * 2u];
    uint b = quads[quad * 2u + 1u];

    ivec3 cell = ivec3(a & 31u, (a >> 5) & 31u, (a >> 10) & 31u);
    int w = int((a >> 15) & 31u) + 1;
    int h = int((a >> 20) & 31u) + 1;
    uint face = (a >> 25) & 7u;
    uint ao = (b >> 16) & 0xFFu;

    // Width/height axes: X faces -> (y, z), Y faces -> (x, z), Z faces -> (x, y).
    ivec3 e1 = face < 2u ? ivec3(0, 1, 0) : ivec3(1, 0, 0);
    ivec3 e2 = face < 4u ? ivec3(0, 0, 1) : ivec3(0, 1, 0);
    bool reversed = face == 1u || face == 2u || face == 5u;
    // Split the quad along the diagonal whose ends are brighter: a single dark corner then
    // shades one triangle instead of smearing along the whole quad.
    uint l00 = corner_light(ao, ivec2(0, 0));
    uint l10 = corner_light(ao, ivec2(1, 0));
    uint l11 = corner_light(ao, ivec2(1, 1));
    uint l01 = corner_light(ao, ivec2(0, 1));
    bool flip = l00 + l11 < l10 + l01;
    ivec2 st = reversed
        ? (flip ? ORDER_B_FLIP[corner_index] : ORDER_B[corner_index])
        : (flip ? ORDER_A_FLIP[corner_index] : ORDER_A[corner_index]);

    ivec3 normal = NORMALS[face];
    ivec3 plane = (face & 1u) == 0u ? normal : ivec3(0); // positive faces lie on the far side of the cell
    ChunkRecord chunk = chunks[gl_InstanceIndex];
    int scale = 1 << (chunk.origin.w - 1); // LOD node: one cell = scale blocks
    ivec3 local = (cell + plane + st.x * w * e1 + st.y * h * e2) * scale;

    // Integer math first (exact far from the world origin), float only for the small remainder.
    ivec3 relative_block = chunk.origin.xyz - frame.camera_block.xyz + local;

    Corner c;
    c.position = vec3(relative_block) - frame.camera_frac.xyz;
    c.normal = vec3(normal);
    c.block = b & 0xFFFFu;
    c.light = float(corner_light(ao, st)) / 3.0;
    c.e1 = e1;
    c.e2 = e2;
    c.st = st;
    c.face = face;
    c.scale = scale;
    c.priority = ao == 0xFFu ? 0 : 1 + ((cell.x + cell.y + cell.z) & 1);
    return c;
}

layout(location = 0) out vec3 v_color;
layout(location = 1) out vec3 v_normal;
layout(location = 2) out vec3 v_position;
layout(location = 3) out float v_light;

void main() {
    Corner c = pull_corner();
    vec3 position = c.position;
    // Merged quads meet other quads in the middle of their edges (T-junctions); rasterization there
    // is not watertight and leaves thin cracks that show what is behind (the sky, or a dark wall
    // below a chunk/LOD border). Push each corner of a horizontal quad outwards by half a pixel
    // ON SCREEN along each quad axis: at grazing angles one block covers far less than a pixel
    // across the view, so a fixed world-space amount is not enough there.
    // Walls do not grow: their top edge would poke above the ground along every step edge.
    vec4 clip = frame.view_proj * vec4(position, 1.0);
    if ((c.face == 2u || c.face == 3u) && clip.w > 0.0) {
        vec2 half_viewport = frame.viewport.xy * 0.5;
        vec3 axes[2] = vec3[2](vec3(c.e1) * float(c.st.x * 2 - 1), vec3(c.e2) * float(c.st.y * 2 - 1));
        vec3 grown = position;
        for (int i = 0; i < 2; i++) {
            vec4 moved = frame.view_proj * vec4(position + axes[i], 1.0);
            if (moved.w > 0.0) {
                // Pixels covered by one block along this axis at this corner.
                float pixels = length((moved.xy / moved.w - clip.xy / clip.w) * half_viewport);
                grown += axes[i] * min(0.5 / max(pixels, 1.0e-4), 0.45 * float(c.scale));
            }
        }
        position = grown;
        clip = frame.view_proj * vec4(position, 1.0);
    }

    // The grown quads overlap their neighbours by up to half a pixel in the same plane, at the same
    // depth. Where the neighbours are shaded differently (an AO corner quad next to a plain merged
    // quad) the winner depended on draw order and flickered: a noisy, "jumping" dark edge along every
    // AO strip. A tiny fixed depth priority decides it instead: AO quads (1x1) in front of merged
    // quads, and neighbouring AO quads by checkerboard parity. Reverse-Z: larger z = closer.
    clip.z *= 1.0 + 4.0e-6 * float(c.priority);
    gl_Position = clip;
    v_color = block_color[c.block].rgb;
    v_normal = c.normal;
    v_position = position;
    v_light = c.light;
}
