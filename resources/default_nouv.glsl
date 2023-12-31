#version 330

#if defined VERTEX_SHADER
uniform mat4 mvp;

in vec3 in_position;

void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
}

#elif defined FRAGMENT_SHADER

out vec4 fragColor;

void main() {
    fragColor = vec4(1.);
}
#endif