#version 460
// Terrain vertex shader (vertex pulling).
// There is no vertex buffer: each quad is 2 uints in the shared geometry pool, and 6 vertices
// (two triangles) are generated from it. gl_VertexIndex / 6 = quad, gl_VertexIndex % 6 = corner.
// gl_InstanceIndex = chunk slot (firstInstance of the indirect command written by cull.comp).

layout(push_constant) uniform Push {
    mat4 view_proj;      // camera rotation + projection (camera at origin)
    ivec4 camera_block;  // camera integer block position relative to the render origin
    vec4 camera_frac;    // xyz: fractional part of camera position; w: fog distance
    vec4 sun_dir;        // xyz: direction to the sun; w: viewport width in pixels
    vec4 fog_color;      // rgb: sky/fog color (linear); w: viewport height in pixels
} pc;

layout(std430, set = 0, binding = 0) readonly buffer Quads { uint quads[]; };
struct ChunkRecord {
    ivec4 origin;  // xyz: chunk corner relative to the render origin, w: 0 = free, else 1 + LOD level
    uvec4 mesh;    // x: first quad in the pool, y/z/w: quad counts per face (used by cull.comp)
};

layout(std430, set = 0, binding = 1) readonly buffer Chunks { ChunkRecord chunks[]; };
layout(std430, set = 0, binding = 2) readonly buffer Colors { vec4 block_color[]; };

layout(location = 0) out vec3 v_color;
layout(location = 1) out vec3 v_normal;
layout(location = 2) out float v_distance;

const ivec3 NORMALS[6] = ivec3[](
    ivec3(1, 0, 0), ivec3(-1, 0, 0),
    ivec3(0, 1, 0), ivec3(0, -1, 0),
    ivec3(0, 0, 1), ivec3(0, 0, -1)
);
// Two triangle orders; which one is used depends on the face so that every face is counter-clockwise
// when seen from outside (back-face culling then removes faces looking away from the camera).
const ivec2 ORDER_A[6] = ivec2[](ivec2(0, 0), ivec2(1, 0), ivec2(1, 1), ivec2(0, 0), ivec2(1, 1), ivec2(0, 1));
const ivec2 ORDER_B[6] = ivec2[](ivec2(0, 0), ivec2(1, 1), ivec2(1, 0), ivec2(0, 0), ivec2(0, 1), ivec2(1, 1));

void main() {
    uint quad = uint(gl_VertexIndex) / 6u;
    uint corner = uint(gl_VertexIndex) % 6u;
    uint a = quads[quad * 2u];
    uint b = quads[quad * 2u + 1u];

    ivec3 cell = ivec3(a & 31u, (a >> 5) & 31u, (a >> 10) & 31u);
    int w = int((a >> 15) & 31u) + 1;
    int h = int((a >> 20) & 31u) + 1;
    uint face = (a >> 25) & 7u;

    // Width/height axes: X faces -> (y, z), Y faces -> (x, z), Z faces -> (x, y).
    ivec3 e1 = face < 2u ? ivec3(0, 1, 0) : ivec3(1, 0, 0);
    ivec3 e2 = face < 4u ? ivec3(0, 0, 1) : ivec3(0, 1, 0);
    bool reversed = face == 1u || face == 2u || face == 5u;
    ivec2 st = reversed ? ORDER_B[corner] : ORDER_A[corner];

    ivec3 normal = NORMALS[face];
    ivec3 plane = (face & 1u) == 0u ? normal : ivec3(0); // positive faces lie on the far side of the cell
    ChunkRecord chunk = chunks[gl_InstanceIndex];
    int scale = 1 << (chunk.origin.w - 1); // LOD node: one cell = scale blocks
    ivec3 local = (cell + plane + st.x * w * e1 + st.y * h * e2) * scale;

    // Integer math first (exact far from the world origin), float only for the small remainder.
    ivec3 relative_block = chunk.origin.xyz - pc.camera_block.xyz + local;
    vec3 position = vec3(relative_block) - pc.camera_frac.xyz;
    // Merged quads meet other quads in the middle of their edges (T-junctions); rasterization there
    // is not watertight and leaves thin cracks that show what is behind (the sky, or a dark wall
    // below a chunk/LOD border). Push each corner of a horizontal quad outwards by half a pixel
    // ON SCREEN along each quad axis: at grazing angles one block covers far less than a pixel
    // across the view, so a fixed world-space amount is not enough there.
    // Walls do not grow: their top edge would poke above the ground along every step edge.
    vec4 clip = pc.view_proj * vec4(position, 1.0);
    if ((face == 2u || face == 3u) && clip.w > 0.0) {
        vec2 half_viewport = vec2(pc.sun_dir.w, pc.fog_color.w) * 0.5;
        vec3 axes[2] = vec3[2](vec3(e1) * float(st.x * 2 - 1), vec3(e2) * float(st.y * 2 - 1));
        vec3 grown = position;
        for (int i = 0; i < 2; i++) {
            vec4 moved = pc.view_proj * vec4(position + axes[i], 1.0);
            if (moved.w > 0.0) {
                // Pixels covered by one block along this axis at this corner.
                float pixels = length((moved.xy / moved.w - clip.xy / clip.w) * half_viewport);
                grown += axes[i] * min(0.5 / max(pixels, 1.0e-4), 0.45 * float(scale));
            }
        }
        position = grown;
        clip = pc.view_proj * vec4(position, 1.0);
    }

    gl_Position = clip;
    v_color = block_color[b & 0xFFFFu].rgb;
    v_normal = vec3(normal);
    v_distance = length(position);
}
