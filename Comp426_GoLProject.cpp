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
static constexpr bool OUTPUT_DEVICE_INFO = true;

// Random number of species (5..10)
std::mt19937 seedGen((unsigned)std::chrono::steady_clock::now().time_since_epoch().count());
std::uniform_int_distribution<int> speciesCountDistribution(5, 10);
const int NUM_SPECIES = (uint8_t)speciesCountDistribution(seedGen);

// Colors for species (RGB each 0-255) - index 0 reserved for dead
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

// OpenCL GPU kernel - computes next game state
const char* cl_gpu_kernel_src = R"CLC(
inline uint lcg_rand_uint(uint state) {
    state = (1103515245u * state + 12345u);
    return state;
}

__kernel void gol_compute_kernel(
    const int width,
    const int height,
    const int num_species,
    const uint frame_seed,
    __global const uchar* grid_in,
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

    grid_out[index] = final_species;
}
)CLC";

// OpenCL CPU kernel - renders grid to RGB
const char* cl_cpu_kernel_src = R"CLC(
__kernel void gol_render_kernel(
    const int width,
    const int height,
    __global const uchar* grid_in,
    __global uchar* rgb_out
){
    int gx = get_global_id(0);
    int gy = get_global_id(1);
    if (gx >= width || gy >= height) return;
    int index = gy * width + gx;
    
    uchar species = grid_in[index];
    
    const uint palette[11] = {
        0xFF000000u, 0xFFA500FFu, 0xFF800080u, 0xFF32CD32u, 0xFFFF00FFu,
        0xFF9000FFu, 0xFF228B22u, 0xFF89CFF0u, 0xFFF5F5DCu, 0xFF00FFFFu, 0xFFFF4500u
    };
    
    uint color = palette[species];
    int p = index * 3;
    rgb_out[p+0] = (color >> 16) & 0xFF;  // R
    rgb_out[p+1] = (color >> 8) & 0xFF;   // G
    rgb_out[p+2] = color & 0xFF;          // B
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
    GLFWwindow* window = glfwCreateWindow(WIN_W, WIN_H, "GOL OpenCL GPU+CPU Pipeline", NULL, NULL);
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
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, GRID_W, GRID_H, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    // Initialize grid randomly
    std::mt19937 rng((unsigned)std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<int> d(0, NUM_SPECIES);
    for (int y = 0; y < GRID_H; ++y) {
        for (int x = 0; x < GRID_W; ++x) {
            grid_cur[idx(x, y)] = (uint8_t)d(rng);
            uint8_t s = grid_cur[idx(x, y)];
            uint32_t color = COLORS[s];
            pixels_cpu[idx(x, y) * 3 + 0] = (color >> 16) & 0xFF; // R
            pixels_cpu[idx(x, y) * 3 + 1] = (color >> 8) & 0xFF;  // G
            pixels_cpu[idx(x, y) * 3 + 2] = color & 0xFF;         // B
        }
    }

    // -------- OpenCL setup --------
    cl_int clerr;
    cl_platform_id platform = nullptr;
    cl_uint num_plat = 0;
    clerr = clGetPlatformIDs(1, &platform, &num_plat);
    check_cl(clerr, "clGetPlatformIDs");

    // Get GPU device
    cl_device_id gpu_device = nullptr;
    cl_uint num_gpu = 0;
    clerr = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &gpu_device, &num_gpu);
    if (clerr != CL_SUCCESS) {
        clerr = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &gpu_device, &num_gpu);
        check_cl(clerr, "clGetDeviceIDs fallback");
    }

    // Get CPU device
    cl_device_id cpu_device = nullptr;
    cl_uint num_cpu = 0;
    clerr = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &cpu_device, &num_cpu);
    if (clerr != CL_SUCCESS) {
        std::cerr << "Warning: No CPU device found, will use GPU for rendering too\n";
        cpu_device = nullptr;
    }

    // Output device info
    if (OUTPUT_DEVICE_INFO) {
        char name[256];
        std::cout << "\n=== GPU Device (Compute) ===\n";
        clGetDeviceInfo(gpu_device, CL_DEVICE_NAME, sizeof(name), name, NULL);
        std::cout << "Device Name: " << name << "\n";
        cl_uint vendorId;
        clGetDeviceInfo(gpu_device, CL_DEVICE_VENDOR_ID, sizeof(vendorId), &vendorId, NULL);
        std::cout << "Vendor ID: " << vendorId << "\n";
        char vendor[256];
        clGetDeviceInfo(gpu_device, CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL);
        std::cout << "Vendor: " << vendor << "\n";

        if (cpu_device) {
            std::cout << "\n=== CPU Device (Render) ===\n";
            clGetDeviceInfo(cpu_device, CL_DEVICE_NAME, sizeof(name), name, NULL);
            std::cout << "Device Name: " << name << "\n";
            clGetDeviceInfo(cpu_device, CL_DEVICE_VENDOR_ID, sizeof(vendorId), &vendorId, NULL);
            std::cout << "Vendor ID: " << vendorId << "\n";
            clGetDeviceInfo(cpu_device, CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL);
            std::cout << "Vendor: " << vendor << "\n";
        }
        std::cout << "\nPipeline: GPU computes next state, CPU renders to RGB\n\n";
    }

    // Create context for GPU
    cl_context_properties props[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
        CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(),
        CL_WGL_HDC_KHR,     (cl_context_properties)wglGetCurrentDC(),
        0
    };

    cl_context gpu_context = clCreateContext(props, 1, &gpu_device, NULL, NULL, &clerr);
    if (clerr != CL_SUCCESS) {
        gpu_context = clCreateContext(NULL, 1, &gpu_device, NULL, NULL, &clerr);
        check_cl(clerr, "clCreateContext GPU fallback");
    }

    cl_command_queue gpu_queue = clCreateCommandQueue(gpu_context, gpu_device, 0, &clerr);
    check_cl(clerr, "clCreateCommandQueue GPU");

    // Create context for CPU
    cl_context cpu_context = nullptr;
    cl_command_queue cpu_queue = nullptr;
    if (cpu_device) {
        cpu_context = clCreateContext(NULL, 1, &cpu_device, NULL, NULL, &clerr);
        check_cl(clerr, "clCreateContext CPU");

        cpu_queue = clCreateCommandQueue(cpu_context, cpu_device, 0, &clerr);
        check_cl(clerr, "clCreateCommandQueue CPU");
    }

    // Build GPU program (compute)
    const char* gpu_src_ptr = cl_gpu_kernel_src;
    cl_program gpu_program = clCreateProgramWithSource(gpu_context, 1, &gpu_src_ptr, nullptr, &clerr);
    check_cl(clerr, "clCreateProgramWithSource GPU");
    clerr = clBuildProgram(gpu_program, 1, &gpu_device, NULL, NULL, NULL);
    if (clerr != CL_SUCCESS) {
        size_t logsz = 0;
        clGetProgramBuildInfo(gpu_program, gpu_device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logsz);
        std::vector<char> log(logsz + 1);
        clGetProgramBuildInfo(gpu_program, gpu_device, CL_PROGRAM_BUILD_LOG, logsz, log.data(), NULL);
        std::cerr << "GPU Build log:\n" << log.data() << "\n";
        check_cl(clerr, "clBuildProgram GPU");
    }

    cl_kernel gpu_kernel = clCreateKernel(gpu_program, "gol_compute_kernel", &clerr);
    check_cl(clerr, "clCreateKernel GPU");

    // Build CPU program (render)
    cl_program cpu_program = nullptr;
    cl_kernel cpu_kernel = nullptr;
    if (cpu_device) {
        const char* cpu_src_ptr = cl_cpu_kernel_src;
        cpu_program = clCreateProgramWithSource(cpu_context, 1, &cpu_src_ptr, nullptr, &clerr);
        check_cl(clerr, "clCreateProgramWithSource CPU");
        clerr = clBuildProgram(cpu_program, 1, &cpu_device, NULL, NULL, NULL);
        if (clerr != CL_SUCCESS) {
            size_t logsz = 0;
            clGetProgramBuildInfo(cpu_program, cpu_device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logsz);
            std::vector<char> log(logsz + 1);
            clGetProgramBuildInfo(cpu_program, cpu_device, CL_PROGRAM_BUILD_LOG, logsz, log.data(), NULL);
            std::cerr << "CPU Build log:\n" << log.data() << "\n";
            check_cl(clerr, "clBuildProgram CPU");
        }

        cpu_kernel = clCreateKernel(cpu_program, "gol_render_kernel", &clerr);
        check_cl(clerr, "clCreateKernel CPU");
    }

    // Create buffers for GPU (compute)
    cl_mem gpu_grid_in = clCreateBuffer(gpu_context, CL_MEM_READ_ONLY, sizeof(uint8_t) * grid_cur.size(), nullptr, &clerr);
    check_cl(clerr, "clCreateBuffer GPU grid_in");

    cl_mem gpu_grid_out = clCreateBuffer(gpu_context, CL_MEM_WRITE_ONLY, sizeof(uint8_t) * grid_cur.size(), nullptr, &clerr);
    check_cl(clerr, "clCreateBuffer GPU grid_out");

    // Create buffers for CPU (render)
    cl_mem cpu_grid_in = nullptr;
    cl_mem cpu_rgb_out = nullptr;
    if (cpu_device) {
        cpu_grid_in = clCreateBuffer(cpu_context, CL_MEM_READ_ONLY, sizeof(uint8_t) * grid_cur.size(), nullptr, &clerr);
        check_cl(clerr, "clCreateBuffer CPU grid_in");

        cpu_rgb_out = clCreateBuffer(cpu_context, CL_MEM_WRITE_ONLY, sizeof(uint8_t) * GRID_W * GRID_H * 3, nullptr, &clerr);
        check_cl(clerr, "clCreateBuffer CPU rgb_out");
    }

    // Set kernel constant args for GPU
    check_cl(clSetKernelArg(gpu_kernel, 0, sizeof(int), &GRID_W), "clSetKernelArg GPU 0");
    check_cl(clSetKernelArg(gpu_kernel, 1, sizeof(int), &GRID_H), "clSetKernelArg GPU 1");
    check_cl(clSetKernelArg(gpu_kernel, 2, sizeof(int), &NUM_SPECIES), "clSetKernelArg GPU 2");

    // Set kernel constant args for CPU
    if (cpu_device) {
        check_cl(clSetKernelArg(cpu_kernel, 0, sizeof(int), &GRID_W), "clSetKernelArg CPU 0");
        check_cl(clSetKernelArg(cpu_kernel, 1, sizeof(int), &GRID_H), "clSetKernelArg CPU 1");
    }

    // GL shader uniform
    glUseProgram(prog);
    glUniform1i(glGetUniformLocation(prog, "uTex"), 0);

    // Main loop
    uint32_t frame = 1;
    while (!glfwWindowShouldClose(window)) {
        auto t0 = std::chrono::high_resolution_clock::now();

        // Upload current grid to both GPU and CPU
        check_cl(clEnqueueWriteBuffer(gpu_queue, gpu_grid_in, CL_FALSE, 0,
            sizeof(uint8_t) * grid_cur.size(), grid_cur.data(), 0, NULL, NULL),
            "clEnqueueWriteBuffer GPU grid_in");

        if (cpu_device) {
            check_cl(clEnqueueWriteBuffer(cpu_queue, cpu_grid_in, CL_FALSE, 0,
                sizeof(uint8_t) * grid_cur.size(), grid_cur.data(), 0, NULL, NULL),
                "clEnqueueWriteBuffer CPU grid_in");
        }

        // GPU: Compute next state
        uint32_t frame_seed = frame * 1640531527u + 123456789u;
        check_cl(clSetKernelArg(gpu_kernel, 3, sizeof(uint32_t), &frame_seed), "clSetKernelArg GPU 3");
        check_cl(clSetKernelArg(gpu_kernel, 4, sizeof(cl_mem), &gpu_grid_in), "clSetKernelArg GPU 4");
        check_cl(clSetKernelArg(gpu_kernel, 5, sizeof(cl_mem), &gpu_grid_out), "clSetKernelArg GPU 5");

        size_t gws[2] = { (size_t)GRID_W, (size_t)GRID_H };
        size_t gpu_lws[2] = { 16, 16 };
        check_cl(clEnqueueNDRangeKernel(gpu_queue, gpu_kernel, 2, NULL, gws, gpu_lws, 0, NULL, NULL),
            "clEnqueueNDRangeKernel GPU");

        // CPU: Render current state to RGB (runs in parallel with GPU compute)
        if (cpu_device) {
            check_cl(clSetKernelArg(cpu_kernel, 2, sizeof(cl_mem), &cpu_grid_in), "clSetKernelArg CPU 2");
            check_cl(clSetKernelArg(cpu_kernel, 3, sizeof(cl_mem), &cpu_rgb_out), "clSetKernelArg CPU 3");

            size_t cpu_lws[2] = { 8, 8 };
            check_cl(clEnqueueNDRangeKernel(cpu_queue, cpu_kernel, 2, NULL, gws, cpu_lws, 0, NULL, NULL),
                "clEnqueueNDRangeKernel CPU");
        }

        // Wait for both to finish
        clFinish(gpu_queue);
        if (cpu_device) {
            clFinish(cpu_queue);
        }

        // Read back results
        std::vector<uint8_t> next_grid(GRID_W * GRID_H);
        check_cl(clEnqueueReadBuffer(gpu_queue, gpu_grid_out, CL_TRUE, 0,
            sizeof(uint8_t) * next_grid.size(), next_grid.data(), 0, NULL, NULL),
            "clEnqueueReadBuffer GPU grid_out");

        if (cpu_device) {
            check_cl(clEnqueueReadBuffer(cpu_queue, cpu_rgb_out, CL_TRUE, 0,
                sizeof(uint8_t) * GRID_W * GRID_H * 3, pixels_cpu.data(), 0, NULL, NULL),
                "clEnqueueReadBuffer CPU rgb_out");
        }
        else {
            // Fallback: render on CPU if no OpenCL CPU device
            for (int i = 0; i < GRID_W * GRID_H; ++i) {
                uint8_t s = grid_cur[i];
                uint32_t color = COLORS[s];
                pixels_cpu[i * 3 + 0] = (color >> 16) & 0xFF; // R
                pixels_cpu[i * 3 + 1] = (color >> 8) & 0xFF;  // G
                pixels_cpu[i * 3 + 2] = color & 0xFF;         // B
            }
        }

        // Upload to GL texture
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, GRID_W, GRID_H, GL_RGB, GL_UNSIGNED_BYTE, pixels_cpu.data());

        // Swap grid buffers
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
        if (elapsed.count() < (1000.0 / 30.0)) {
            Sleep((DWORD)((1000.0 / 31.0) - elapsed.count()));
            t1 = std::chrono::high_resolution_clock::now();
            elapsed = t1 - t0;
        }
        double fps = 1000.0 / (elapsed.count() + 1e-6);
        if ((frame & 31) == 0) {
            std::cout << "frame " << frame << " fps ~ " << fps
                << " num_species=" << NUM_SPECIES
                << " (GPU:compute, CPU:render)\n";
        }
    }

    // Clean up GPU resources
    clReleaseMemObject(gpu_grid_in);
    clReleaseMemObject(gpu_grid_out);
    clReleaseKernel(gpu_kernel);
    clReleaseProgram(gpu_program);
    clReleaseCommandQueue(gpu_queue);
    clReleaseContext(gpu_context);

    // Clean up CPU resources
    if (cpu_device) {
        clReleaseMemObject(cpu_grid_in);
        clReleaseMemObject(cpu_rgb_out);
        clReleaseKernel(cpu_kernel);
        clReleaseProgram(cpu_program);
        clReleaseCommandQueue(cpu_queue);
        clReleaseContext(cpu_context);
    }

    // Clean up GL resources
    glDeleteTextures(1, &tex);
    glDeleteProgram(prog);
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &ebo);
    glDeleteVertexArrays(1, &vao);
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}