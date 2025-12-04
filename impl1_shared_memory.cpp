#include <hip/hip_runtime.h>
#include <math.h>
#include <stdio.h>
#include <sys/time.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <sys/stat.h>
#include <sys/types.h>
#include <iomanip>
#include <sstream>

static float alpha = 1.0f;
static int num_iterations = 100;

// ==================== GPU KERNELS ====================


// Kernel to compute image derivatives Ix, Iy, It
__global__ void compute_derivatives(float* I1, float* I2, float* Ix, float* Iy, float* It, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) return;

    int idx = y * width + x;
    Ix[idx] = (I1[idx + 1] - I1[idx - 1] + I2[idx + 1] - I2[idx - 1]) * 0.25f;
    Iy[idx] = (I1[idx + width] - I1[idx - width] + I2[idx + width] - I2[idx - width]) * 0.25f;
    It[idx] = (I2[idx] - I1[idx]);
}


// Kernel to perform one iteration of Horn-Schunck update
__global__ void horn_schunck_iteration(float* U_old, float* V_old, float* U_new, float* V_new,
                                        float* Ix, float* Iy, float* It, int width, int height, float alpha2) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) return;

    int idx = y * width + x;
    float U_avg = (U_old[idx - 1] + U_old[idx + 1] + U_old[idx - width] + U_old[idx + width]) * 0.25f;
    float V_avg = (V_old[idx - 1] + V_old[idx + 1] + V_old[idx - width] + V_old[idx + width]) * 0.25f;

    float numerator = Ix[idx] * U_avg + Iy[idx] * V_avg + It[idx];
    float denominator = alpha2 + Ix[idx] * Ix[idx] + Iy[idx] * Iy[idx];
    U_new[idx] = U_avg - (Ix[idx] * numerator) / denominator;
    V_new[idx] = V_avg - (Iy[idx] * numerator) / denominator;
}

// ==================== Other CPU functions (just copied from impl1.cpp) ====================

// function to output grayscale bin files from input video and return number of files created
int video_to_grayscale_bins(const char* input_video, const char* output_folder) {
    mkdir(output_folder, 0777);

    cv::VideoCapture cap(input_video);
    if (!cap.isOpened()) {
        printf("Error opening video file\n");
        return 0;
    }

    int frame_count = 0;
    cv::Mat frame, gray_frame;
    while (cap.read(frame)) {
        cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
        gray_frame.convertTo(gray_frame, CV_32F, 1.0 / 255.0);

        char filename[256];
        snprintf(filename, sizeof(filename), "%s/frame_%04d.bin", output_folder, frame_count);

        FILE* file = fopen(filename, "wb");
        if (!file) {
            printf("Error opening %s for write\n", filename);
            break;
        }
        fwrite(gray_frame.ptr<float>(), sizeof(float), gray_frame.rows * gray_frame.cols, file);
        fclose(file);

        frame_count++;
    }
    cap.release();

    printf("Actually extracted %d frames.\n", frame_count);
    return frame_count;
}


// constant function that takes in the input.mp4 video and outputs the number of frames, width, and height
void get_video_properties(const char* filename,int& width, int& height) {
    cv::VideoCapture cap(filename);
    if (!cap.isOpened()) {
        printf("Error opening video file\n");
        return;
    }
    width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    cap.release();
}

// Helper function to load a frame from a binary file
std::vector<float> load_frame(const char* filename, int width, int height) {
    std::vector<float> frame(width * height);
    FILE* file = fopen(filename, "rb");
    fread(frame.data(), sizeof(float), width * height, file);
    fclose(file);
    return frame;
}

// Helper function to generate zero-padded filenames
std::string padded(int number) {
    std::ostringstream ss;
    ss << std::setw(4) << std::setfill('0') << number;
    return ss.str();
}

// ==================== MAIN ====================

int main() {
    int width, height;

    get_video_properties("input.mp4", width, height);

    int num_frames = video_to_grayscale_bins("input.mp4", "frames");

    printf("Video properties - Frames: %d, Width: %d, Height: %d\n", num_frames, width, height);

    // Create output folder
    mkdir("flow", 0777);

    // GPU memory allocation
    int size = width * height * sizeof(float);
    float *d_I1, *d_I2, *d_Ix, *d_Iy, *d_It;
    float *d_U_old, *d_U_new, *d_V_old, *d_V_new;
    hipMalloc(&d_I1, size);
    hipMalloc(&d_I2, size);
    hipMalloc(&d_Ix, size);
    hipMalloc(&d_Iy, size);
    hipMalloc(&d_It, size);
    hipMalloc(&d_U_old, size);
    hipMalloc(&d_U_new, size);
    hipMalloc(&d_V_old, size);
    hipMalloc(&d_V_new, size);

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    float alpha2 = alpha * alpha;

    // Host arrays for result
    std::vector<float> U(width * height);
    std::vector<float> V(width * height);
    std::vector<float> flow(width * height * 2);

    // Load all frames
    std::vector<std::vector<float>> frames(num_frames);
    for (int i = 0; i < num_frames; ++i) {
        std::string filename = "frames/frame_" + padded(i) + ".bin";
        frames[i] = load_frame(filename.c_str(), width, height);
    }

    // Process each frame pair on GPU
    for (int i = 0; i < num_frames - 1; ++i) {
        auto& I1 = frames[i];
        auto& I2 = frames[i+1];

        // Copy frames to GPU
        hipMemcpy(d_I1, I1.data(), size, hipMemcpyHostToDevice);
        hipMemcpy(d_I2, I2.data(), size, hipMemcpyHostToDevice);

        // Compute derivatives (once per frame pair)
        hipLaunchKernelGGL(compute_derivatives, grid, block, 0, 0,
                           d_I1, d_I2, d_Ix, d_Iy, d_It, width, height);

        // Initialize flow to zero
        hipMemset(d_U_old, 0, size);
        hipMemset(d_V_old, 0, size);

        // Iterative refinement of flow estimates
        for (int iter = 0; iter < num_iterations; ++iter) {
            hipLaunchKernelGGL(horn_schunck_iteration, grid, block, 0, 0,
                               d_U_old, d_V_old, d_U_new, d_V_new,
                               d_Ix, d_Iy, d_It, width, height, alpha2);
            std::swap(d_U_old, d_U_new);
            std::swap(d_V_old, d_V_new);
        }

        // Copy result back to CPU
        hipMemcpy(U.data(), d_U_old, size, hipMemcpyDeviceToHost);
        hipMemcpy(V.data(), d_V_old, size, hipMemcpyDeviceToHost);

        // Combine U and V into a single output vector
        for (int j = 0; j < width * height; ++j) {
            flow[2 * j] = U[j];
            flow[2 * j + 1] = V[j];
        }

        // Save flow to binary file within the "flow" folder
        FILE* f = fopen(("flow/frame_" + padded(i) + ".bin").c_str(), "wb");
        fwrite(flow.data(), sizeof(float), flow.size(), f);
        fclose(f);
    }

    // Cleanup
    hipFree(d_I1);
    hipFree(d_I2);
    hipFree(d_Ix);
    hipFree(d_Iy);
    hipFree(d_It);
    hipFree(d_U_old);
    hipFree(d_U_new);
    hipFree(d_V_old);
    hipFree(d_V_new);

    return 0;
}
