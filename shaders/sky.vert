#version 460
// Full-screen triangle for the sky, at the far plane (reverse-Z: depth 0).
// The pipeline draws it after the terrain with "greater or equal" depth test, so only pixels
// where nothing was drawn (depth still 0) run the sky shader.

layout(location = 0) out vec2 v_ndc;

void main() {
    vec2 ndc = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2)) * 2.0 - 1.0;
    v_ndc = ndc;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
