#version 460
// Terrain vertex shader (vertex pulling).
// There is no vertex buffer: each quad is 2 uints in the shared geometry pool, and 6 vertices
// (two triangles) are generated from it. gl_VertexIndex / 6 = quad, gl_VertexIndex % 6 = corner.
// gl_InstanceIndex = chunk slot (passed as firstInstance), used to fetch the chunk origin.

layout(push_constant) uniform Push {
    mat4 view_proj;      // camera rotation + projection (camera at origin)
    ivec4 camera_block;  // camera integer block position relative to the render origin
    vec4 camera_frac;    // xyz: fractional part of camera position; w: fog distance
    vec4 sun_dir;        // xyz: direction to the sun
    vec4 fog_color;      // rgb: sky/fog color (linear)
} pc;

layout(std430, set = 0, binding = 0) readonly buffer Quads { uint quads[]; };
layout(std430, set = 0, binding = 1) readonly buffer Chunks { ivec4 chunk_origin[]; };
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
    ivec3 local = cell + plane + st.x * w * e1 + st.y * h * e2;

    // Integer math first (exact far from the world origin), float only for the small remainder.
    ivec3 relative_block = chunk_origin[gl_InstanceIndex].xyz - pc.camera_block.xyz + local;
    vec3 position = vec3(relative_block) - pc.camera_frac.xyz;

    gl_Position = pc.view_proj * vec4(position, 1.0);
    v_color = block_color[b & 0xFFFFu].rgb;
    v_normal = vec3(normal);
    v_distance = length(position);
}
