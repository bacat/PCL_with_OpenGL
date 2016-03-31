#ifndef VERYNAIVESPHERE_HPP_INCLUDED
#define VERYNAIVESPHERE_HPP_INCLUDED

#include <vector>

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

/*
    Later, I should generalize the movement parts and collision detection part
    of this class, so I can use it more generally.
    (ie. create a new templated class for moving all type of 3D shapes)
*/

class VeryNaiveSphere
{
    std::vector<GLfloat> vertices; // point vector of my sphere

    int radius{0};

    glm::vec3 direction = glm::vec3(0.f);
    glm::vec3 pos = glm::vec3(0.f); // actual position

    GLfloat speed{0.f};

    // This is for preventing the sphere from getting another collision before the bounce.
    // More comes later
    bool isCollision{false};

    GLuint VBO, VAO; // for effective visualization, I'm gonna use VertexBufferObject

public:
    // constructor for the sphere
    VeryNaiveSphere(int init_radius, float init_speed, glm::vec3 init_pos)
    {
        radius = init_radius;
        pos = init_pos;
        speed = init_speed;

        float red = 1.f / 180.f;

        for(float Yaw = 0.f; Yaw < 360.f; Yaw += 2.f)
        {
            for(float Pitch = 0.f; Pitch < 180.f; Pitch += 2.f)
            {
                // Spherical coordinate system again.
                // This is a very naive visualization of the sphere, but for my tests, it will be OK.
                vertices.emplace_back(radius * cos(glm::radians(Yaw)) * sin(glm::radians(Pitch)));
                vertices.emplace_back(radius * sin(glm::radians(Yaw)) * sin(glm::radians(Pitch)));
                vertices.emplace_back(radius * cos(glm::radians(Pitch)));

                vertices.emplace_back(red * Pitch); // To get the color according to the sphere point.
                vertices.emplace_back(0.5f);
                vertices.emplace_back(1.f);
            }
        }

    }

    // Initialize my sphere's VBO and give my points to the graphics card's memory
    void init_sphere()
    {
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
    }

    void draw(GLuint MatrixID_model, glm::mat4 Model) const
    {
        // These parameters are my world's matrices
        // I get them, and modify it to get the right position my ball.
        // (The other solution is to use another shader.)
        Model = glm::translate(Model, pos);

        glUniformMatrix4fv(MatrixID_model, 1, GL_FALSE, &Model[0][0]);

        glBindVertexArray(VAO);
        glDrawArrays(GL_POINTS, 0, vertices.size() / 6);
        glBindVertexArray(0);
    }
    // Remember that the dot product, which returns a scalar value, can be used in projecting a vector onto another axis.
    // To project V onto N, the formula is (V dot N)*N.
    glm::vec3 getResponseVectorFromNormals(glm::vec3 direction, glm::vec3 normal)
    {
        return glm::normalize(-2 * glm::dot(direction, normal) * normal + direction);
    }

    // This is just for pre-testing the case of big-speed movement.
    glm::vec3 getAverageOfNormals(std::vector<glm::vec3> normals)
    {
        glm::vec3 sumOfNormals(0.f);

        for(auto normal : normals)
        {
            sumOfNormals += normal;
        }

        return sumOfNormals / (float)normals.size();
    }

    // Calculate velocity, handle the collision, and calculate the new position.
    void move(GLfloat deltaTime, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, pcl::octree::OctreePointCloudSearch<pcl::PointXYZRGB> &octree)
    {
        GLfloat velocity = speed * deltaTime;

        handleBounce(cloud, octree);

        pos += direction * velocity;
    }

    // To handle bounce.
    void handleBounce(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, pcl::octree::OctreePointCloudSearch<pcl::PointXYZRGB> &octree)
    {
        pcl::PointXYZRGB searchPoint;

        // This is the point, where I want to search the nearest points inside the radius.
        // This should be the center of the sphere.
        searchPoint.x = pos[0];
        searchPoint.y = pos[1];
        searchPoint.z = pos[2];

        std::vector<int> pointIdxRadiusSearch; // index of the cloud points within the radius
        std::vector<float> pointRadiusSquaredDistance; // distance of the indexed points from the reference point

        // collecting points within the sphere
        octree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);

        if(pointIdxRadiusSearch.size() == 0)
        {
            /*
             This part of code was written to prevent the sphere from another bounce effect
             before the end of the previous bounce effect.
             When the sphere bounced, and _still_ the points from the cloud are in my sphere,
             I don't wanna calculate a new direction vector again.

             Good to know that this is special case, with special response vector and special speed.
            */
            if(isCollision) isCollision = false;
            return;
        }

        if(isCollision) return;

        // After the previous bounce there are no more points inside of the sphere:
        isCollision = true;

        pointIdxRadiusSearch.clear();
        pointRadiusSquaredDistance.clear();

        // Now I want to find the closest point.
        // If the speed is not degenerated, we can be sure, that the closest point was the first point the sphere reached.
        // (If the speed is degenerated, well.... this case is on my TODO list. :D)
        octree.nearestKSearch(searchPoint, 1, pointIdxRadiusSearch, pointRadiusSquaredDistance);

        auto idx = pointIdxRadiusSearch[0];

        // The new searchpont is the closest point from the cloud we found.
        searchPoint.x = cloud->points[idx].x;
        searchPoint.y = cloud->points[idx].y;
        searchPoint.z = cloud->points[idx].z;

        int num_of_referencePoints = 30; // for calculating surface normal

        // clear the vectors again
        pointIdxRadiusSearch.clear();
        pointRadiusSquaredDistance.clear();

        // We want to find the closest points to compute the surface normal.
        octree.nearestKSearch(searchPoint, num_of_referencePoints, pointIdxRadiusSearch, pointRadiusSquaredDistance);

        Eigen::Vector4f vec4;
        float curvature{0.f};

        // And finally, I compute the normal:
        pcl::computePointNormal(*cloud, pointIdxRadiusSearch, vec4, curvature);

        glm::vec3 normal = glm::vec3(vec4[0], vec4[1], vec4[2]);

        direction = getResponseVectorFromNormals(direction, normal);
    }

    // Some functions to handle my sphere's direction and speed easier.
    void changeDirToX_pos() { direction = glm::vec3(1.f, 0.f, 0.f); }
    void changeDirToX_neg() { direction = glm::vec3(-1.f, 0.f, 0.f); }
    void changeDirToY_pos() { direction = glm::vec3(0.f, 1.f, 0.f); }
    void changeDirToY_neg() { direction = glm::vec3(0.f, -1.f, 0.f); }
    void changeDirToZ_pos() { direction = glm::vec3(0.f, 0.f, 1.f); }
    void changeDirToZ_neg() { direction = glm::vec3(0.f, 0.f, -1.f); }

    void increaseSpeed(float m_speed) { speed += 1000; }
    void decreaseSpeed(float m_speed) { speed -= 1000; }
};

#endif // VERYNAIVESPHERE_HPP_INCLUDED
