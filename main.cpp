#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/io/io.h>
#include <pcl/impl/point_types.hpp>

#define GLEW_STATIC
#include <GL/glew.h>

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Shader.hpp"
#include "Camera.hpp"
#include "VeryNaiveSphere.hpp"

#include <future>

pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

float min_x{0}, max_x{0}, min_y{0}, max_y{0}, min_z{0}, max_z{0};

// Function prototypes
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void Do_Camera_movement();
void Do_Sphere_movement(VeryNaiveSphere &mySphere, pcl::octree::OctreePointCloudSearch<pcl::PointXYZRGB> &octree);

// Window dimensions
constexpr GLuint WIDTH = 800, HEIGHT = 600;

// my matrices for visualization on the screen
glm::mat4 ModelView = glm::mat4(1.0f);
glm::mat4 Proj = glm::mat4(1.0f);

float cam_near{0.1f}, cam_far{90000.f}; // define clipping plane

// For camera, and key statements
Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));
bool keys[1024]; // to store if a key is actually pressed
GLfloat lastX = WIDTH / 2.f, lastY = HEIGHT / 2.f;
bool firstMouse = true;

GLfloat deltaTime = 0.0f;
GLfloat lastFrame = 0.0f;

// The MAIN function, from here we start the application and run the main loop
int main()
{
    ///*************************************************************************************///
    ///******************************** PCL LOADING PART ***********************************///
    ///*************************************************************************************///

    // (This part will be separated later!)

    // Later, this will be according to the command line argument
    // Now I test it, with "fovam2a_bin_compressed.pcd"
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB> ("fovam2a_bin_compressed.pcd", *cloud) == -1) // load the file
    {
        PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
        return (-1);
    }
    std::cout << "Loaded " << cloud->width * cloud->height << " data points from test_pcd.pcd with the following fields: " << std::endl;


    std::cout << cloud->width << std::endl;
    std::cout << cloud->height << std::endl;

    // Set the initial minimum value of each coordinate:
    min_x = cloud->points[0].x;
    min_y = cloud->points[0].y;
    min_z = cloud->points[0].z;

    // Calculate the minimum value of each coordinate:
    for (size_t i = 0; i < cloud->points.size (); ++i)
    {
        if(cloud->points[i].x < min_x) min_x = cloud->points[i].x;
        if(cloud->points[i].y < min_y) min_y = cloud->points[i].y;
        if(cloud->points[i].z < min_z) min_z = cloud->points[i].z;
    }

    // Transform the cloud to the origin of its coordinate system, for easier handling of the cloud data.
    // This part of the code should be removed later. (Just helped me at the beginning)
    for(size_t i = 0; i < cloud->points.size (); ++i)
    {
        cloud->points[i].x -= min_x;
        cloud->points[i].y -= min_y;
        cloud->points[i].z -= min_z;
    }

    ///*************************************************************************************///
    ///******************************** GLFW INIT PART ***********************************///
    ///*************************************************************************************///

    // Init GLFW
    glfwInit();

    // Set all the required options for GLFW
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

    // Create a GLFWwindow object that we can use for GLFW's functions
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "PCL with OpenGL", nullptr, nullptr);
    glfwMakeContextCurrent(window);

    // Set the required callback functions
    glfwSetKeyCallback(window, key_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // Options
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // This is how GLEW knows that it should use a modern approach for retrieving function pointers and extensions
    glewExperimental = GL_TRUE;

    // Initialize GLEW to setup the OpenGL Function pointers
    glewInit();

    // Define the viewport dimensions
    glViewport(0, 0, WIDTH, HEIGHT);

    glEnable(GL_DEPTH_TEST);
    glPointSize(1.3f);

    ///*************************************************************************************///
    ///******************************** VBO LOADING PART ***********************************///
    ///*************************************************************************************///

    Shader ourShader("shader.vs", "shader.frag"); // load the shaders

    std::size_t v_size = cloud->points.size() * 6; // size of my points
    // (all of them contains 6 member)

    // I will fill up this vector with all the data from cloud->points
    std::vector<GLfloat> vertices(v_size);

    // for efficient data reading, I start a new thread for reading the half of my data.
    std::future<void> result( std::async([&]()
    {
        for(size_t i = 1; i < cloud->points.size (); i+=2)
        {
            size_t num = (i * 6);

            vertices[num + 0] = cloud->points[i].x;
            vertices[num + 1] = cloud->points[i].y;
            vertices[num + 2] = cloud->points[i].z;

            vertices[num + 3] = (float)cloud->points[i].r / 256.f;
            vertices[num + 4] = (float)cloud->points[i].g / 256.f;
            vertices[num + 5] = (float)cloud->points[i].b / 256.f;
        }

    }));

    // another half of my points
    for(size_t i = 0; i < cloud->points.size (); i+=2)
    {
        size_t num = (i * 6);

        vertices[num + 0] = cloud->points[i].x;
        vertices[num + 1] = cloud->points[i].y;
        vertices[num + 2] = cloud->points[i].z;

        vertices[num + 3] = (float)cloud->points[i].r / 256.f;
        vertices[num + 4] = (float)cloud->points[i].g / 256.f;
        vertices[num + 5] = (float)cloud->points[i].b / 256.f;
    }

    result.get(); // wait for the other thread
    // Now, I've filled up my vector!

    std::cout << "*****!!!DONE! (read)!!!*****" << std::endl;

    /// Create the VBO from the data:
    GLuint VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    // Bind the Vertex Array Object first, then bind and set the vertex buffer(s) and the attribute pointer(s).
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * vertices.size(), vertices.data(), GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (GLvoid*)0);
    glEnableVertexAttribArray(0);

    // Color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0); // Unbind VAO

    std::cout << "*****!!!DONE! (VBO load)!!!*****" << std::endl;

    ourShader.Use();

    /// Load transformation matrices
    GLuint MatrixID_modelview = glGetUniformLocation(ourShader.Program, "modelview");
    GLuint MatrixID_proj = glGetUniformLocation(ourShader.Program, "projection");

    // Send our transformations to the currently bound shader
    glUniformMatrix4fv(MatrixID_modelview, 1, GL_FALSE, &ModelView[0][0]);
    glUniformMatrix4fv(MatrixID_proj, 1, GL_FALSE, &Proj[0][0]);

    vertices.clear();
    //cloud.reset();

    VeryNaiveSphere mySphere(500, 500, glm::vec3(10000.f, 10000.f, 10000.f));
    mySphere.init_sphere();

    float resolution = 128.0f; // for the leaf level of the octree

    pcl::octree::OctreePointCloudSearch<pcl::PointXYZRGB> octree (resolution);

    // Fill up my tree:
    octree.setInputCloud (cloud);
    octree.addPointsFromInputCloud();

    ///*************************************************************************************///
    ///******************************** MAIN LOOP PART ***********************************///
    ///*************************************************************************************///

    std::cout << "*****!!!LOOP!!!*****" << std::endl;
    // Main loop
    while (!glfwWindowShouldClose(window))
    {
        // Set frame time
        GLfloat currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // Check if any events have been activiated (key pressed, mouse moved etc.) and call corresponding response functions
        glfwPollEvents();
        Do_Camera_movement();
        Do_Sphere_movement(mySphere, octree);

        // Render
        // Clear the colorbuffer and the depthbuffer
        glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Draw
        ourShader.Use();

        Proj = glm::perspective(camera.Zoom, (float)WIDTH / (float)HEIGHT, cam_near, cam_far);

        ModelView = camera.GetViewMatrix();

        glUniformMatrix4fv(MatrixID_proj, 1, GL_FALSE, &Proj[0][0]);
        glUniformMatrix4fv(MatrixID_modelview, 1, GL_FALSE, &ModelView[0][0]);

        glBindVertexArray(VAO);
        glDrawArrays(GL_POINTS, 0, v_size / 6);
        glBindVertexArray(0);

        mySphere.draw(MatrixID_modelview, ModelView);

        // Swap the screen buffers
        glfwSwapBuffers(window);
    }
    // Properly de-allocate all resources once they've outlived their purpose
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);

    // Terminate GLFW, clearing any resources allocated by GLFW.
    glfwTerminate();
    return 0;
}

    ///*************************************************************************************///
    ///******************************** FUNC IMPLEMETATION ***********************************///
    ///*************************************************************************************///

// Moves/alters the camera positions based on user input
void Do_Camera_movement()
{
    // Camera controls
    if(keys[GLFW_KEY_W])
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if(keys[GLFW_KEY_S])
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if(keys[GLFW_KEY_A])
        camera.ProcessKeyboard(LEFT, deltaTime);
    if(keys[GLFW_KEY_D])
        camera.ProcessKeyboard(RIGHT, deltaTime);
}

void Do_Sphere_movement(VeryNaiveSphere &mySphere, pcl::octree::OctreePointCloudSearch<pcl::PointXYZRGB> &octree)
{
    // Sphere conrols
    // Set pre-defined directions:
    if(keys[GLFW_KEY_T])
        mySphere.changeDirToX_neg();
    if(keys[GLFW_KEY_Z])
        mySphere.changeDirToX_pos();
    if(keys[GLFW_KEY_G])
        mySphere.changeDirToY_neg();
    if(keys[GLFW_KEY_H])
        mySphere.changeDirToY_pos();
    if(keys[GLFW_KEY_B])
        mySphere.changeDirToZ_neg();
    if(keys[GLFW_KEY_N])
        mySphere.changeDirToZ_pos();

    // Move the sphere:
    if(keys[GLFW_KEY_M])
        mySphere.move(deltaTime, cloud, octree);
}

// This is called whenever a key is pressed/released via GLFW
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode)
{
    if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);

    if (key >= 0 && key < 1024)
    {
        if(action == GLFW_PRESS)
            keys[key] = true;
        else if(action == GLFW_RELEASE)
            keys[key] = false;
    }
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    if(firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    GLfloat xoffset = xpos - lastX;
    GLfloat yoffset = lastY - ypos;  // Reversed, since y-coordinates go from bottom-left corner

    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}


void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(yoffset);
}
