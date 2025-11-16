#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <thread>

using namespace std::chrono_literals;

static constexpr int W = 1024;
static constexpr int H = 768;
static constexpr int TARGET_FPS = 500;
static constexpr bool TRACK_FPS = true;

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

//__constant__ int dx[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };
//__constant__ int dy[8] = { -1,-1,-1, 0, 0, 1, 1, 1 };

struct World {
    uint8_t* currentGrid;
    uint8_t* nextGrid;
    int w, h, species;
};

std::mt19937 seedGen((unsigned)std::time(nullptr));

//__global__ void computeNextGenerationKernel(uint8_t* cur, uint8_t* nxt,
//    int W, int H, int speciesCount,
//    unsigned long seed) {
//    int x = blockIdx.x * blockDim.x + threadIdx.x;
//    int y = blockIdx.y * blockDim.y + threadIdx.y;
//    if (x >= W || y >= H) return;
//
//    const int idx = y * W + x;
//    const uint8_t currentCell = cur[idx];
//
//    int neighborCount[10] = { 0 };
//    for (int n = 0; n < 8; ++n) {
//        int nx = (x + dx[n] + W) % W; // wraps
//        int ny = (y + dy[n] + H) % H;
//
//        uint8_t neighbor = cur[ny * W + nx];
//        if (neighbor) {
//            neighborCount[neighbor - 1]++;
//        }
//    }
//
//
//    if (currentCell) {
//        // Alive condition, if 2 or 3 same species neighbors
//        int speciesIndex = currentCell - 1;
//        int sameSpecies = neighborCount[speciesIndex];
//        nxt[idx] = (sameSpecies == 2 || sameSpecies == 3) ? currentCell : 0;
//    }
//    else {
//        int candidates[10];
//        int count = 0;
//        for (int s = 0; s < speciesCount; ++s)
//            if (neighborCount[s] == 3)
//                candidates[count++] = s;
//
//        if (count == 1) {
//            // Simple birth
//            nxt[idx] = (uint8_t)(candidates[0] + 1);
//        }
//        else if (count > 1) {
//            // Random birth (>1)
//            curandState rng;
//            curand_init(seed + idx, 0, 0, &rng);
//            nxt[idx] = (uint8_t)(candidates[(int)(curand_uniform(&rng) * count)] + 1);
//        }
//        else {
//            // Dead condition
//            nxt[idx] = 0;
//        }
//    }
//}

void computeNextGeneration(World& world, uint32_t seed) {
    //dim3 grid((world.w + block.x - 1) / block.x, (world.h + block.y - 1) / block.y);
    //dim3 block(16, 16);
    //computeNextGenerationKernel << <grid, block >> > (world.currentGrid, world.nextGrid, world.w, world.h, world.species, seed);
    //
    //cudaError_t err = cudaGetLastError();
    //if (err != cudaSuccess) {
    //    std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
    //}
    //
    //cudaDeviceSynchronize();
}

static GLuint glHelpCreateTexture(int w, int h) {
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); // pixels stay crisp
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    // Allocate storage
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    return tex;
}

void speciesGridToPixels(uint8_t* grid, std::vector<uint32_t>& pixels, const std::vector<uint32_t>& palette, int N) {
    std::vector<uint8_t> thisGrid(N);
    //cudaMemcpy(thisGrid.data(), grid, N * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    pixels.resize(N);
    for (int i = 0; i < N; ++i) pixels[i] = palette[thisGrid[i]];
}

static void glHelpDrawFullscreenTexture(GLuint tex, int w, int h) {
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, w, h, 0, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, tex);

    glBegin(GL_QUADS);
    glTexCoord2f(0.f, 0.f); glVertex2f(0.f, 0.f);
    glTexCoord2f(1.f, 0.f); glVertex2f((float)w, 0.f);
    glTexCoord2f(1.f, 1.f); glVertex2f((float)w, (float)h);
    glTexCoord2f(0.f, 1.f); glVertex2f(0.f, (float)h);
    glEnd();

    glDisable(GL_TEXTURE_2D);
}

static void checkCudaError() {
    //cudaError_t err = cudaGetLastError();
    //if (err != cudaSuccess) {
    //    std::cerr << cudaGetErrorString(err) << std::endl;
    //}
    //cudaDeviceSynchronize();
}

int main() {
    if (!glfwInit()) return 1;
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow* win = glfwCreateWindow(W, H, "CUDA Game of Life", nullptr, nullptr);
    if (!win) return 1;
    glfwMakeContextCurrent(win);

    std::uniform_int_distribution<int> speciesCountDistribution(5, 10);
    int speciesCount = speciesCountDistribution(seedGen);

    size_t N = W * H;
    World world{ nullptr,
                 nullptr,
                 W, H, speciesCount
    };

    std::vector<uint8_t> initialGrid(N);
    std::uniform_int_distribution<int> speciesPick(1, speciesCount);
    for (auto& c : initialGrid) c = (uint8_t)speciesPick(seedGen);

    // Initialize the memory on the GPU to have the grids.
    //cudaMalloc(&world.currentGrid, N * sizeof(uint8_t));
    //cudaMalloc(&world.nextGrid, N * sizeof(uint8_t));
    //cudaMemcpy(world.currentGrid, initialGrid.data(), N * sizeof(uint8_t), cudaMemcpyHostToDevice);

    std::vector<uint32_t> pixels;
    GLuint texture = glHelpCreateTexture(W, H);

    // FPS Tracking, unneeded
    auto nextTick = std::chrono::steady_clock::now();
    auto lastTime = nextTick;
    int frames = 0;

    // mr loop
    while (!glfwWindowShouldClose(win)) {
        auto now = std::chrono::steady_clock::now();
        if (now < nextTick)
            std::this_thread::sleep_for(nextTick - now);
        nextTick += std::chrono::duration_cast<std::chrono::steady_clock::duration>(TICK_INTERVAL);

        // 1024 * 768 = 786,432
        //dim3 block(16, 16); // a group of threads, 16x16 is 256 threads.
        //dim3 grid((world.w + block.x - 1) / block.x,    // 1024/16 = 64
        //    (world.h + block.y - 1) / block.y);   // 768 /16 = 48
        //    // 256 * 64 * 48 = 786,432
        //
        //
        //computeNextGenerationKernel << <grid, block >> > (world.currentGrid, world.nextGrid,
        //    world.w, world.h, world.species,
        //    seedGen());
        //checkCudaError();

        std::swap(world.currentGrid, world.nextGrid);

        speciesGridToPixels(world.currentGrid, pixels, COLORS, N);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
        glClear(GL_COLOR_BUFFER_BIT);
        glHelpDrawFullscreenTexture(texture, W, H);
        glfwSwapBuffers(win);
        glfwPollEvents();

        // FPS Tracking, also unneeded
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

    glfwTerminate();
    return 0;
}
