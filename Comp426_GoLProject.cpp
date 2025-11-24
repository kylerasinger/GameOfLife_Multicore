#include <windows.h>
#include <glew.h>
#include <GLFW/glfw3.h>
#include <CL/cl.h>
#include <CL/cl_gl.h>
#include <gl/GL.h>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>
#include <cstring>

static constexpr int W = 1024;
static constexpr int H = 768;
static constexpr int TARGET_FPS = 500;
static constexpr bool TRACK_FPS = true;
static constexpr bool OUTPUT_DEVICE_INFO = true;

std::mt19937 seedGen((unsigned)std::chrono::steady_clock::now().time_since_epoch().count());
std::uniform_int_distribution<int> speciesCountDistribution(5, 10);
const int SPECIES_COUNT = (uint8_t)speciesCountDistribution(seedGen);

static const std::vector<uint32_t> COLORS = {
    0xFF000000u, // dead
    0xFFA500FFu,
    0xFF800080u,
    0xFF32CD32u,
    0xFFFF00FFu,
    0xFF9000FFu,
    0xFF228B22u,
    0xFF89CFF0u,
    0xFFF5F5DCu,
    0xFF00FFFFu,
    0xFFFF4500u
};

static constexpr auto TICK_INTERVAL = std::chrono::duration<double>(1.0 / TARGET_FPS);

int index(int x, int y) { return y * W + x; }

struct World {
    std::vector<uint8_t> currentGrid;
    std::vector<uint8_t> nextGrid;
    int w, h, species;
};
 
// Computes frame N+1
const char* clComputeKernelSourceString = R"CLC(
inline uint lcg_rand_uint(uint state) {
    state = (1103515245u * state + 12345u);
    return state;
}

__kernel void gol_logic_compute_kernel(
    const int width,
    const int height,
    const int numSpecies,
    const uint frameSeed,
    __global const uchar* gridIn,
    __global uchar* grid_out
){
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= width || y >= height) return;
    
    const int idx = y * width + x;
    const uchar currentCell = gridIn[idx];
    
    // Direction offsets for 8 neighbors
    const int dx[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };
    const int dy[8] = { -1, -1, -1, 0, 0, 1, 1, 1 };
    
    int neighborCount[10] = { 0 };
    
    for (int n = 0; n < 8; ++n) {
        int nx = (x + dx[n] + width) % width; // wraps
        int ny = (y + dy[n] + height) % height;
        uchar neighbor = gridIn[ny * width + nx];
        if (neighbor) {
            neighborCount[neighbor - 1]++;
        }
    }
    
    if (currentCell) {
        // Alive condition, if 2 or 3 same species neighbors
        int speciesIndex = currentCell - 1;
        int sameSpecies = neighborCount[speciesIndex];
        grid_out[idx] = (sameSpecies == 2 || sameSpecies == 3) ? currentCell : 0;
    }
    else {
        int candidates[10];
        int count = 0;
        for (int s = 0; s < numSpecies; ++s) {
            if (neighborCount[s] == 3) {
                candidates[count++] = s;
            }
        }
        
        if (count == 1) {
            // Simple birth
            grid_out[idx] = (uchar)(candidates[0] + 1);
        }
        else if (count > 1) {
            // Random birth (>1)
            uint state = (uint)(x * y * frameSeed); // More random tie breaker
            state = lcg_rand_uint(state);
            int selected = state % count;
            grid_out[idx] = (uchar)(candidates[selected] + 1);
        }
        else {
            // Dead condition
            grid_out[idx] = 0;
        }
    }
}
)CLC";

// Renders frame N.
const char* clRenderKernelSourceString = R"CLC(
__kernel void gol_render_kernel(
    const int width,
    const int height,
    __global const uchar* gridIn,
    __global uchar* rgbaPbo
){
    int gx = get_global_id(0);
    int gy = get_global_id(1);
    if (gx >= width || gy >= height) return;
    int index = gy * width + gx;
    
    uchar species = gridIn[index];
    
    const uint palette[11] = {
        0xFF000000u, 0xFFA500FFu, 0xFF800080u, 0xFF32CD32u, 0xFFFF00FFu,
        0xFF9000FFu, 0xFF228B22u, 0xFF89CFF0u, 0xFFF5F5DCu, 0xFF00FFFFu, 0xFFFF4500u
    };
    
    uint color = palette[species];
    int p = index * 4;
    rgbaPbo[p+0] = (color >> 16) & 0xFF;  // R
    rgbaPbo[p+1] = (color >> 8) & 0xFF;   // G
    rgbaPbo[p+2] = color & 0xFF;          // B
    rgbaPbo[p+3] = 0xFF;                  // A
}
)CLC";

// Helper method to output any openCl errors, easier debugging.
static void checkClHelper(cl_int err, const char* msg) {
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL Error: " << err << " : " << msg << "\n";
        exit(EXIT_FAILURE);
    }
}

int main() {
    std::cout << "Random species count: " << SPECIES_COUNT << "\n\n";
    World world;
    world.w = W;
    world.h = H;
    world.species = SPECIES_COUNT;
    world.currentGrid.resize(W * H);
    world.nextGrid.resize(W * H);

    // Initialize grid randomly
    std::mt19937 rng((unsigned)std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<int> d(0, world.species);
    for (int y = 0; y < world.h; ++y) {
        for (int x = 0; x < world.w; ++x) {
            world.currentGrid[index(x, y)] = (uint8_t)d(rng);
        }
    }

    if (!glfwInit()) return 1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE); // This allows us to use old OpenGL calls.
    GLFWwindow* win = glfwCreateWindow(W, H, "OpenCL CPU and GPU Game of Life", nullptr, nullptr);
    if (!win) return 2;
    glfwMakeContextCurrent(win);

    glfwSwapInterval(0);
    if (glewInit() != GLEW_OK) return 3;

    // Set up for orthographic rendering (2D Rendering)
    glMatrixMode(GL_PROJECTION);
    glOrtho(0, W, H, 0, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glDisable(GL_DEPTH_TEST);

    // Create GL texture for display
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    
    // Set up OpenCL
    cl_int clerr;
    cl_platform_id platform = nullptr;
    cl_uint num_plat = 0;
    cl_device_id gpuDevice = nullptr;
    cl_uint num_gpu = 0;
    checkClHelper(clGetPlatformIDs(1, &platform, &num_plat), "clGetPlatformIDs");
    checkClHelper(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &gpuDevice, &num_gpu), "clGetDeviceIDs");

    // Check for cl_khr_gl_sharing extension support
    size_t extensionsSize;
    clGetDeviceInfo(gpuDevice, CL_DEVICE_EXTENSIONS, 0, nullptr, &extensionsSize);
    std::vector<char> extensions(extensionsSize);
    clGetDeviceInfo(gpuDevice, CL_DEVICE_EXTENSIONS, extensionsSize, extensions.data(), nullptr);
    std::string extensionsStr(extensions.data());
    if (extensionsStr.find("cl_khr_gl_sharing") == std::string::npos) {
        std::cerr << "Error: cl_khr_gl_sharing extension not supported!\n";
        return 4;
    }

    // Create OpenGL Pixel Buffer Object (PBO) for shared memory
    GLuint pbo;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, W * H * 4 * sizeof(uint8_t), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // Output device info
    if (OUTPUT_DEVICE_INFO) {
        char name[256];
        std::cout << "--===+ GPU Device (Compute) +===--\n";
        clGetDeviceInfo(gpuDevice, CL_DEVICE_NAME, sizeof(name), name, NULL);
        std::cout << "Device Name: " << name << "\n";
        cl_uint vendorId;
        clGetDeviceInfo(gpuDevice, CL_DEVICE_VENDOR_ID, sizeof(vendorId), &vendorId, NULL);
        std::cout << "Vendor ID: " << vendorId << "\n";
        char vendor[256];
        clGetDeviceInfo(gpuDevice, CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL);
        std::cout << "Vendor: " << vendor << "\n";
        std::cout << "Info is from openCl clGetDeviceInfo()\n\n";
    }

    // Create context with GL interop, this is what allows OpenCl to call OpenGl. Taken from slides
    cl_context_properties properties[] = {
        CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(),
        CL_WGL_HDC_KHR,     (cl_context_properties)wglGetCurrentDC(),
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
        0
    };

    cl_context context = clCreateContext(properties, 1, &gpuDevice, NULL, NULL, &clerr);
    checkClHelper(clerr, "clCreateContext");

    cl_command_queue commandQueue = clCreateCommandQueue(context, gpuDevice, 0, &clerr);
    checkClHelper(clerr, "clCreateCommandQueue");

    // Build compute program
    const char* computeKernelSource = clComputeKernelSourceString;
    cl_program computeProgram = clCreateProgramWithSource(context, 1, &computeKernelSource, nullptr, &clerr);
    checkClHelper(clerr, "clCreateProgramWithSource");
    checkClHelper(clBuildProgram(computeProgram, 1, &gpuDevice, NULL, NULL, NULL), "clBuildProgram");
    cl_kernel computeKernel = clCreateKernel(computeProgram, "gol_logic_compute_kernel", &clerr);
    checkClHelper(clerr, "clCreateKernel");

    // Build render program
    const char* renderKernelSource = clRenderKernelSourceString;
    cl_program renderProgram = clCreateProgramWithSource(context, 1, &renderKernelSource, nullptr, &clerr);
    checkClHelper(clerr, "clCreateProgramWithSource");
    checkClHelper(clBuildProgram(renderProgram, 1, &gpuDevice, NULL, NULL, NULL), "clBuildProgram");
    cl_kernel renderKernel = clCreateKernel(renderProgram, "gol_render_kernel", &clerr);
    checkClHelper(clerr, "clCreateKernel");

    // Create OpenCL mem for grid
    cl_mem clCurrentGrid = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(uint8_t) * world.currentGrid.size(), nullptr, &clerr);
    checkClHelper(clerr, "clCreateBuffer");
    cl_mem clNextGrid = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(uint8_t) * world.nextGrid.size(), nullptr, &clerr);
    checkClHelper(clerr, "clCreateBuffer");

    // Create OpenCL buffer from OpenGL PBO
    cl_mem clRgbaBufferPBO = clCreateFromGLBuffer(context, CL_MEM_WRITE_ONLY, pbo, &clerr);
    checkClHelper(clerr, "clCreateFromGLBuffer");

    checkClHelper(clSetKernelArg(computeKernel, 0, sizeof(int), &world.w), "clSetKernelArg compute 0");
    checkClHelper(clSetKernelArg(computeKernel, 1, sizeof(int), &world.h), "clSetKernelArg compute 1");
    checkClHelper(clSetKernelArg(computeKernel, 2, sizeof(int), &world.species), "clSetKernelArg compute 2");
    checkClHelper(clSetKernelArg(renderKernel, 0, sizeof(int), &world.w), "clSetKernelArg render 0");
    checkClHelper(clSetKernelArg(renderKernel, 1, sizeof(int), &world.h), "clSetKernelArg render 1");

    auto nextTick = std::chrono::steady_clock::now();
    auto lastTime = nextTick;
    int frame = 0;
    int frames = 0;

    // mr loop
    while (!glfwWindowShouldClose(win)) {
        // Compute kernel computes frame N+1.
        // Render kernel renders frame N.

        auto now = std::chrono::steady_clock::now();
        if (now < nextTick)
        {
            auto diff = nextTick - now;
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(diff).count();
            Sleep((DWORD)ms);
        }
        nextTick += std::chrono::duration_cast<std::chrono::steady_clock::duration>(TICK_INTERVAL);
        ++frame;

        // Upload current grid to GPU
        checkClHelper(clEnqueueWriteBuffer(commandQueue, clCurrentGrid, CL_FALSE, 0,
                                           sizeof(uint8_t) * world.currentGrid.size(), 
                                           world.currentGrid.data(), 0, NULL, NULL),
                      "clEnqueueWriteBuffer");

        // Compute next state
        uint32_t frameSeed = frame * SPECIES_COUNT; // This makes it random based on the frame count and species count
        checkClHelper(clSetKernelArg(computeKernel, 3, sizeof(uint32_t), &frameSeed), "clSetKernelArg compute 3");
        checkClHelper(clSetKernelArg(computeKernel, 4, sizeof(cl_mem), &clCurrentGrid), "clSetKernelArg compute 4");
        checkClHelper(clSetKernelArg(computeKernel, 5, sizeof(cl_mem), &clNextGrid), "clSetKernelArg compute 5");

        // Same thread count and work-groups as the CUDA implementation. 
        size_t threadCount[2] = { (size_t)world.w, (size_t)world.h };
        size_t groupSize[2] = { 16, 16 };
        checkClHelper(clEnqueueNDRangeKernel(commandQueue, computeKernel, 2, NULL, threadCount, 
                                             groupSize, 0, NULL, NULL),
                      "clEnqueueNDRangeKernel");
        glFlush();

        // Acquire shared GL object for OpenCL use
        checkClHelper(clEnqueueAcquireGLObjects(commandQueue, 1, &clRgbaBufferPBO, 0, NULL, NULL),
                      "clEnqueueAcquireGLObjects");

        // Render current grid to shared PBO
        checkClHelper(clSetKernelArg(renderKernel, 2, sizeof(cl_mem), &clCurrentGrid), "clSetKernelArg render 2");
        checkClHelper(clSetKernelArg(renderKernel, 3, sizeof(cl_mem), &clRgbaBufferPBO), "clSetKernelArg render 3");

        checkClHelper(clEnqueueNDRangeKernel(commandQueue, renderKernel, 2, NULL,
                                             threadCount, groupSize, 0, NULL, NULL),
                      "clEnqueueNDRangeKernel");

        // Release shared GL object back to OpenGL
        checkClHelper(clEnqueueReleaseGLObjects(commandQueue, 1, &clRgbaBufferPBO, 0, NULL, NULL),
                      "clEnqueueReleaseGLObjects");

        // Wait for OpenCL to finish
        clFinish(commandQueue);

        // Read back next grid state
        checkClHelper(clEnqueueReadBuffer(commandQueue, clNextGrid, CL_TRUE, 0,
                                          sizeof(uint8_t) * world.nextGrid.size(), world.nextGrid.data(),
                                          0, NULL, NULL),
                      "clEnqueueReadBuffer grid_out");

        // Update texture from PBO
        glBindTexture(GL_TEXTURE_2D, tex);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, world.w, world.h, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // Swap grid buffers
        world.currentGrid.swap(world.nextGrid);

        glClear(GL_COLOR_BUFFER_BIT);
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, tex);

        glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f); glVertex2f(0.0f, 0.0f);
        glTexCoord2f(1.0f, 0.0f); glVertex2f((float)world.w, 0.0f);
        glTexCoord2f(1.0f, 1.0f); glVertex2f((float)world.w, (float)world.h);
        glTexCoord2f(0.0f, 1.0f); glVertex2f(0.0f, (float)world.h);
        glEnd();
        glDisable(GL_TEXTURE_2D);

        glfwSwapBuffers(win);
        glfwPollEvents();

        if (glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS) break;

        if (TRACK_FPS) {
            frames++;
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - lastTime);
            if (elapsed.count() >= 1) {
                std::cout << "FPS: " << frames / elapsed.count() << std::endl;
                frames = 0;
                lastTime = now;
            }
        }
    }

    clReleaseMemObject(clCurrentGrid);
    clReleaseMemObject(clNextGrid);
    clReleaseMemObject(clRgbaBufferPBO);
    clReleaseKernel(computeKernel);
    clReleaseKernel(renderKernel);
    clReleaseProgram(computeProgram);
    clReleaseProgram(renderProgram);
    clReleaseCommandQueue(commandQueue);
    clReleaseContext(context);
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &tex);
    glfwDestroyWindow(win);
    glfwTerminate();

    return 0;
}