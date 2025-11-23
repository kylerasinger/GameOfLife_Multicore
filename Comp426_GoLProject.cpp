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

// Window / grid size
static constexpr int WIN_W = 1024;
static constexpr int WIN_H = 768;
static constexpr int GRID_W = WIN_W;
static constexpr int GRID_H = WIN_H;
static constexpr bool OUTPUT_DEVICE_INFO = false;

// Random number of species (5..10)
std::mt19937 seedGen((unsigned)std::chrono::steady_clock::now().time_since_epoch().count());
std::uniform_int_distribution<int> speciesCountDistribution(5, 10);
const int NUM_SPECIES = (uint8_t)speciesCountDistribution(seedGen);

// Colors for species (RGB each 0-255) - index 0 reserved for dead
const uint8_t SPECIES_COLORS[11][3] = {
    {0, 0, 0},        // dead
    {255, 0, 0},      // species1
    {0, 255, 0},      // species2
    {0, 0, 255},      // species3
    {255, 255, 0},    // species4
    {255, 0, 255},    // species5
    {0, 255, 255},    // species6
    {255, 165, 0},    // species7
    {128, 0, 128},    // species8
    {192, 192, 192},  // species9
    {255, 255, 255}   // species10
};

int idx(int x, int y) { return y * GRID_W + x; }

// CPU-side buffers
std::vector<uint8_t> grid_cur(GRID_W* GRID_H);
std::vector<uint8_t> pixels_cpu(GRID_W* GRID_H * 3);

// Shader sources
const char* vshader_src = R"(
#version 330 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aUV;
out vec2 vUV;
void main(){
    vUV = aUV;
    gl_Position = vec4(aPos,0.0,1.0);
}
)";
const char* fshader_src = R"(
#version 330 core
in vec2 vUV;
out vec4 FragColor;
uniform sampler2D uTex;
void main(){
    FragColor = texture(uTex, vUV);
}
)";

// OpenCL kernel source
const char* cl_kernel_src = R"CLC(
__constant sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

inline uint lcg_rand_uint(uint state) {
    state = (1103515245u * state + 12345u);
    return state;
}

__kernel void gol_kernel(
    const int width,
    const int height,
    const int num_species,
    const uint frame_seed,
    __global const uchar* grid_in,
    __global uchar* rgb_out,
    __global uchar* grid_out
){
    int gx = get_global_id(0);
    int gy = get_global_id(1);
    if (gx >= width || gy >= height) return;
    int index = gy * width + gx;
    uchar cur = grid_in[index];

    uchar counts[11];
    for (int i=0;i<=num_species;i++) counts[i] = 0;

    int x0 = (gx>0? gx-1 : 0);
    int x1 = (gx<width-1? gx+1 : width-1);
    int y0 = (gy>0? gy-1 : 0);
    int y1 = (gy<height-1? gy+1 : height-1);

    for (int yy=y0; yy<=y1; ++yy) {
        int row = yy * width;
        for (int xx=x0; xx<=x1; ++xx) {
            if (xx==gx && yy==gy) continue;
            uchar s = grid_in[row + xx];
            counts[s] = counts[s] + (uchar)1;
        }
    }

    uchar candidates[11];
    uchar cand_cnt = 0;
    for (int s=1; s<=num_species; ++s) {
        uchar neighbors = counts[s];
        uchar cur_alive = (cur == (uchar)s) ? 1 : 0;
        uchar next_alive = 0;
        if (cur_alive) {
            next_alive = (neighbors == 2 || neighbors == 3) ? 1 : 0;
        } else {
            next_alive = (neighbors == 3) ? 1 : 0;
        }
        if (next_alive) {
            candidates[cand_cnt++] = (uchar)s;
        }
    }

    uchar final_species = 0;
    if (cand_cnt > 0) {
        uint state = (uint)(gx * 73856093u ^ gy * 19349663u ^ frame_seed);
        state = lcg_rand_uint(state);
        uint pick = state % cand_cnt;
        final_species = candidates[pick];
    }

    // write grid_out
    grid_out[index] = final_species;

    // write rgb_out
    const uchar palette[11][3] = {
        {0,0,0}, {255,0,0}, {0,255,0}, {0,0,255}, {255,255,0},
        {255,0,255}, {0,255,255}, {255,165,0}, {128,0,128}, {192,192,192}, {255,255,255}
    };
    int p = index*3;
    rgb_out[p+0] = palette[final_species][0];
    rgb_out[p+1] = palette[final_species][1];
    rgb_out[p+2] = palette[final_species][2];
}
)CLC";

static void check_cl(cl_int err, const char* msg) {
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL Error (" << err << "): " << msg << std::endl;
        exit(EXIT_FAILURE);
    }
}

GLuint compile_shader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char buf[1024];
        glGetShaderInfoLog(s, 1024, NULL, buf);
        std::cerr << "Shader compile error: " << buf << std::endl;
    }
    return s;
}

int main() {
    // GLFW + GL init
    if (!glfwInit()) {
        std::cerr << "GLFW init failed\n";
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(WIN_W, WIN_H, "GOL OpenCL Windows Iris Xe", NULL, NULL);
    if (!window) {
        std::cerr << "Window creation failed\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);
    if (glewInit() != GLEW_OK) {
        std::cerr << "GLEW init failed\n";
        return -1;
    }

    // Setup quad
    float quadVerts[] = {
        -1.f,-1.f, 0.f,0.f,
         1.f,-1.f, 1.f,0.f,
         1.f, 1.f, 1.f,1.f,
        -1.f, 1.f, 0.f,1.f
    };
    unsigned int quadIdx[] = { 0,1,2,2,3,0 };
    GLuint vao, vbo, ebo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVerts), quadVerts, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(quadIdx), quadIdx, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);

    // Compile GL shader
    GLuint vs = compile_shader(GL_VERTEX_SHADER, vshader_src);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fshader_src);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    {
        GLint ok2; glGetProgramiv(prog, GL_LINK_STATUS, &ok2);
        if (!ok2) {
            char buf[1024]; glGetProgramInfoLog(prog, 1024, NULL, buf);
            std::cerr << "Link error: " << buf << "\n";
        }
    }
    glDeleteShader(vs); glDeleteShader(fs);

    // Create texture
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    // allocate RGBA8
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, GRID_W, GRID_H, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    // Initialize grid randomly
    std::mt19937 rng((unsigned)std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<int> d(0, NUM_SPECIES);
    for (int y = 0; y < GRID_H; ++y) {
        for (int x = 0; x < GRID_W; ++x) {
            grid_cur[idx(x, y)] = (uint8_t)d(rng);
            uint8_t s = grid_cur[idx(x, y)];
            pixels_cpu[idx(x, y) * 3 + 0] = SPECIES_COLORS[s][0];
            pixels_cpu[idx(x, y) * 3 + 1] = SPECIES_COLORS[s][1];
            pixels_cpu[idx(x, y) * 3 + 2] = SPECIES_COLORS[s][2];
        }
    }

    // -------- OpenCL setup --------
    cl_int clerr;
    cl_platform_id platform = nullptr;
    cl_uint num_plat = 0;
    clerr = clGetPlatformIDs(1, &platform, &num_plat);
    check_cl(clerr, "clGetPlatformIDs");

    cl_device_id device = nullptr;
    cl_uint num_dev = 0;
    clerr = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num_dev);

    // Output device info
    if (OUTPUT_DEVICE_INFO) {
        char name[256];
        clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, NULL);
        std::cout << "Device Name: " << name << "\n";
        cl_uint vendorId;
        clGetDeviceInfo(device, CL_DEVICE_VENDOR_ID, sizeof(vendorId), &vendorId, NULL);
        std::cout << "Vendor ID: " << vendorId << "\n";
        char vendor[256];
        clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL);
        std::cout << "Vendor: " << vendor << "\n";
    }

    if (clerr != CL_SUCCESS) {
        clerr = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, &num_dev);
        check_cl(clerr, "clGetDeviceIDs fallback");
    }

    // Prepare context properties for WGL sharing
    cl_context_properties props[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
        CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(),
        CL_WGL_HDC_KHR,     (cl_context_properties)wglGetCurrentDC(),
        0
    };

    cl_context clContext = clCreateContext(props, 1, &device, NULL, NULL, &clerr);
    if (clerr != CL_SUCCESS) {
        // fallback: create normal context
        clContext = clCreateContext(NULL, 1, &device, NULL, NULL, &clerr);
        check_cl(clerr, "clCreateContext fallback");
    }

    //cl_command_queue_properties properties[] =
    //{
    //    CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(),
    //    CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(),
    //    CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
    //};


    cl_command_queue clQueue = clCreateCommandQueue(
        clContext,
        device,
        CL_QUEUE_PROFILING_ENABLE,
        &clerr
    );

    check_cl(clerr, "clCreateCommandQueue");

    // Build program
    const char* src_ptr = cl_kernel_src;
    cl_program program = clCreateProgramWithSource(clContext, 1, &src_ptr, nullptr, &clerr);
    check_cl(clerr, "clCreateProgramWithSource");
    clerr = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (clerr != CL_SUCCESS) {
        size_t logsz = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logsz);
        std::vector<char> log(logsz + 1);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logsz, log.data(), NULL);
        std::cerr << "Build log:\n" << log.data() << "\n";
        check_cl(clerr, "clBuildProgram");
    }

    cl_kernel kernel = clCreateKernel(program, "gol_kernel", &clerr);
    check_cl(clerr, "clCreateKernel");

    // Create buffers
    cl_mem d_grid_in = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uint8_t) * grid_cur.size(), grid_cur.data(), &clerr);
    check_cl(clerr, "clCreateBuffer grid_in");

    cl_mem d_grid_out = clCreateBuffer(clContext, CL_MEM_READ_WRITE, sizeof(uint8_t) * grid_cur.size(), nullptr, &clerr);
    check_cl(clerr, "clCreateBuffer grid_out");

    cl_mem d_rgb_out = clCreateBuffer(clContext, CL_MEM_WRITE_ONLY, sizeof(uint8_t) * GRID_W * GRID_H * 3, nullptr, &clerr);
    check_cl(clerr, "clCreateBuffer rgb_out");

    // Set kernel constant args (width, height, species)
    check_cl(clSetKernelArg(kernel, 0, sizeof(int), &GRID_W), "clSetKernelArg 0");
    check_cl(clSetKernelArg(kernel, 1, sizeof(int), &GRID_H), "clSetKernelArg 1");
    check_cl(clSetKernelArg(kernel, 2, sizeof(int), &NUM_SPECIES), "clSetKernelArg 2");

    // GL shader uniform
    glUseProgram(prog);
    glUniform1i(glGetUniformLocation(prog, "uTex"), 0);

    // Main loop
    uint32_t frame = 1;
    while (!glfwWindowShouldClose(window)) {
        auto t0 = std::chrono::high_resolution_clock::now();

        // Write current grid to device
        check_cl(clEnqueueWriteBuffer(clQueue, d_grid_in, CL_TRUE, 0,
            sizeof(uint8_t) * grid_cur.size(), grid_cur.data(), 0, NULL, NULL),
            "clEnqueueWriteBuffer grid_in");

        // Set per-frame args: frame seed, grid_in/out, rgb_out
        uint32_t frame_seed = frame * 1640531527u + 123456789u;
        check_cl(clSetKernelArg(kernel, 3, sizeof(uint32_t), &frame_seed), "clSetKernelArg 3");
        check_cl(clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_grid_in), "clSetKernelArg 4");
        check_cl(clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_rgb_out), "clSetKernelArg 5");
        check_cl(clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_grid_out), "clSetKernelArg 6");

        size_t gws[2] = { (size_t)GRID_W, (size_t)GRID_H };
        size_t lws[2] = { 16, 16 }; // tuneable for Intel Iris Xe
        check_cl(clEnqueueNDRangeKernel(clQueue, kernel, 2, NULL, gws, lws, 0, NULL, NULL), "clEnqueueNDRangeKernel");

        clFinish(clQueue);

        // Read back rgb_out to host
        check_cl(clEnqueueReadBuffer(clQueue, d_rgb_out, CL_TRUE, 0,
            sizeof(uint8_t) * GRID_W * GRID_H * 3, pixels_cpu.data(), 0, NULL, NULL), "clEnqueueReadBuffer rgb_out");

        // Upload to GL texture (RGBA8 but we only filled RGB bytes; we'll set A=255 via GL)
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, GRID_W, GRID_H, GL_RGB, GL_UNSIGNED_BYTE, pixels_cpu.data());

        // Swap grid buffers on host: copy grid_out -> grid_cur
        std::vector<uint8_t> next_grid(GRID_W * GRID_H);
        check_cl(clEnqueueReadBuffer(clQueue, d_grid_out, CL_TRUE, 0,
            sizeof(uint8_t) * next_grid.size(), next_grid.data(), 0, NULL, NULL),
            "clEnqueueReadBuffer grid_out");
        grid_cur.swap(next_grid);

        // Render
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(prog);
        glBindVertexArray(vao);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        glfwSwapBuffers(window);
        glfwPollEvents();

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) break;

        ++frame;
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = t1 - t0;
        // Limit to 30 FPS
        if (elapsed.count() < (1000.0 / 30.0)) {
            Sleep((DWORD)((1000.0 / 31.0) - elapsed.count()));
            t1 = std::chrono::high_resolution_clock::now();
            elapsed = t1 - t0;
        }
        double fps = 1000.0 / (elapsed.count() + 1e-6);
        if ((frame & 31) == 0) {
            std::cout << "frame " << frame << " fps ~ " << fps << " num_species=" << NUM_SPECIES << "\n";
        }
    }

    // Clean up
    clReleaseMemObject(d_grid_in);
    clReleaseMemObject(d_grid_out);
    clReleaseMemObject(d_rgb_out);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(clQueue);
    clReleaseContext(clContext);

    glDeleteTextures(1, &tex);
    glDeleteProgram(prog);
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &ebo);
    glDeleteVertexArrays(1, &vao);
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
