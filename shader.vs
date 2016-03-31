#version 330 core
    layout (location = 0) in vec3 position;
    layout (location = 1) in vec3 color;
    out vec3 ourColor;

    uniform mat4 modelview;
    uniform mat4 projection;

    void main()
    {
	gl_Position = projection * modelview *  vec4(position, 1.0);
    	ourColor = color;
    }
