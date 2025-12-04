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
#include <algorithm>

// regularization parameter and number of iterations
static float alpha = 5.0f;
static int num_iterations = 20;

// ==================== GPU KERNELS ====================

// Kernel to compute sobel image derivatives Ix, Iy, It
// Sobel derivatives in x and y directions, and temporal derivative
// This type of derivative is better for optical flow than simple finite differences
// since it smooths noise while computing gradients
__global__ void compute_derivatives(const float* __restrict__ I1,
                                    const float* __restrict__ I2,
                                    float* __restrict__ Ix,
                                    float* __restrict__ Iy,
                                    float* __restrict__ It,
                                    int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // need a 1-pixel border for 3x3 Sobel
    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) return;

    int idx = y * width + x;
    int idx_up = (y - 1) * width;
    int idx_mid = y * width;
    int idx_down = (y + 1) * width;

    // Use only I1 for spatial derivatives
    const float* I = I1;

    // 3x3 neighborhood:
    float I_ul = I[idx_up + (x - 1)];
    float I_um = I[idx_up +  x    ];
    float I_ur = I[idx_up + (x + 1)];

    float I_ml = I[idx_mid + (x - 1)];
    float I_mm = I[idx_mid + x];
    float I_mr = I[idx_mid + (x + 1)];

    float I_ll = I[idx_down + (x - 1)];
    float I_lm = I[idx_down + x];
    float I_lr = I[idx_down + (x + 1)];

    // Sobel Gx (horizontal derivative)
    // [-1  0  1
    //  -2  0  2
    //  -1  0  1]
    float gx = (-I_ul + I_ur -2.0f * I_ml + 2.0f * I_mr -I_ll + I_lr);

    // Sobel Gy (vertical derivative)
    // [-1 -2 -1
    //   0  0  0
    //   1  2  1]
    float gy = (-I_ul - 2.0f * I_um - I_ur + I_ll + 2.0f * I_lm + I_lr);

    // scaling by 8 for Sobel normalization
    Ix[idx] = gx * (1.0f / 8.0f);
    Iy[idx] = gy * (1.0f / 8.0f);

    // Temporal derivative
    It[idx] = I2[idx] - I1[idx];
}

// Kernel to perform one iteration of Horn-Schunck update
__global__ void horn_schunck_iteration(float* U_old, float* V_old, float* U_new, float* V_new,
                                        float* Ix, float* Iy, float* It, int width, int height, float alpha2) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) return;

    int idx = y * width + x;

    // indices of neighboring pixels
    int xm1 = x - 1;
    int xp1 = x + 1;
    int ym1 = y - 1;
    int yp1 = y + 1;

    // weighted 8-neighbor average
    float U_avg =
        (U_old[(ym1 * width) + xm1] * (1.0f/12.0f) +
         U_old[(ym1 * width) + x] * (1.0f/6.0f) +
         U_old[(ym1 * width) + xp1] * (1.0f/12.0f) +
         U_old[(y * width) + xm1] * (1.0f/6.0f) +
         U_old[(y * width) + xp1] * (1.0f/6.0f) +
         U_old[(yp1 * width) + xm1] * (1.0f/12.0f) +
         U_old[(yp1 * width) + x] * (1.0f/6.0f) +
         U_old[(yp1 * width) + xp1] * (1.0f/12.0f));

    float V_avg =
        (V_old[(ym1 * width) + xm1] * (1.0f/12.0f) +
         V_old[(ym1 * width) + x] * (1.0f/6.0f) +
         V_old[(ym1 * width) + xp1] * (1.0f/12.0f) +
         V_old[(y * width) + xm1] * (1.0f/6.0f) +
         V_old[(y * width) + xp1] * (1.0f/6.0f) +
         V_old[(yp1 * width) + xm1] * (1.0f/12.0f) +
         V_old[(yp1 * width) + x] * (1.0f/6.0f) +
         V_old[(yp1 * width) + xp1] * (1.0f/12.0f));

    float numerator = Ix[idx] * U_avg + Iy[idx] * V_avg + It[idx];
    float denominator = alpha2 + Ix[idx] * Ix[idx] + Iy[idx] * Iy[idx];
    U_new[idx] = U_avg - (Ix[idx] * numerator) / denominator;
    V_new[idx] = V_avg - (Iy[idx] * numerator) / denominator;
}

// ==================== Other CPU functions (just copied from impl1.cpp) ====================

// constant function that takes in the input.mp4 video and outputs the fps, width, and height
void get_video_properties(const char* filename,int& width, int& height, float& fps) {
    cv::VideoCapture cap(filename);
    if (!cap.isOpened()) {
        printf("Error opening video file\n");
        return;
    }
    width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    fps = static_cast<float>(cap.get(cv::CAP_PROP_FPS));
    cap.release();
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
    float fps;

    get_video_properties("input.mp4", width, height, fps);

    cv::VideoCapture cap("input.mp4");
    if (!cap.isOpened()) {
        printf("Error opening input.mp4\n");
        return -1;
    }

    int num_frames = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);

    printf("Video properties - Frames: %d, Width: %d, Height: %d, FPS: %.2f\n", num_frames, width, height, fps);

    // create output video
    cv::VideoWriter video_out(
        "optical_flow_output.mp4",
        cv::VideoWriter::fourcc('a','v','c','1'), // H.264 encoding
        fps,
        cv::Size(width, height)
    );

    if (!video_out.isOpened()) {
        printf("ERROR: Could not open video writer!\n");
        return -1;
    }

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

    // Read all frames into a vector
    cv::Mat frame1_bgr, frame2_bgr;
    cv::Mat I1_mat, I2_mat;
    cv::Mat mag(height, width, CV_32F);
    cv::Mat ang(height, width, CV_32F);

    // Read first frame
    cap >> frame1_bgr;
    if (frame1_bgr.empty()) return 0;

    cv::cvtColor(frame1_bgr, I1_mat, cv::COLOR_BGR2GRAY);
    I1_mat.convertTo(I1_mat, CV_32F, 1.0 / 255.0);

    // Process each frame pair on GPU
    for (int i = 0; i < num_frames - 1; ++i) {
        cap >> frame2_bgr;
        if (frame2_bgr.empty()) break;

        cv::cvtColor(frame2_bgr, I2_mat, cv::COLOR_BGR2GRAY);
        I2_mat.convertTo(I2_mat, CV_32F, 1.0 / 255.0);

        // Copy frames to GPU
        hipMemcpy(d_I1, I1_mat.ptr<float>(), size, hipMemcpyHostToDevice);
        hipMemcpy(d_I2, I2_mat.ptr<float>(), size, hipMemcpyHostToDevice);

        // Compute derivatives (once per frame pair)
        compute_derivatives<<<grid, block>>>(d_I1, d_I2, d_Ix, d_Iy, d_It, width, height);

        // Initialize flow to zero
        hipMemset(d_U_old, 0, size);
        hipMemset(d_V_old, 0, size);

        // Iterative refinement of flow estimates
        for (int iter = 0; iter < num_iterations; ++iter) {
            horn_schunck_iteration<<<grid, block>>>(d_U_old, d_V_old, d_U_new, d_V_new,
                                                     d_Ix, d_Iy, d_It, width, height, alpha2);
            std::swap(d_U_old, d_U_new);
            std::swap(d_V_old, d_V_new);
        }

        // Copy result back to CPU
        hipMemcpy(U.data(), d_U_old, size, hipMemcpyDeviceToHost);
        hipMemcpy(V.data(), d_V_old, size, hipMemcpyDeviceToHost);

        // build mag and ang matrices
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = y * width + x;
                float u = U[idx];
                float v = V[idx];
                mag.at<float>(y,x) = sqrtf(u*u + v*v);
                ang.at<float>(y,x) = atan2f(v, u);
            }
        }

        // normalize magnitudes to [0,1]
        cv::Mat mag_norm;
        cv::normalize(mag, mag_norm, 0.0f, 1.0f, cv::NORM_MINMAX);

        cv::Mat hsv(height, width, CV_32FC3);

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float angle = ang.at<float>(y,x) * 180.0f / M_PI;
                if (angle < 0) angle += 360.0f;

                float magnitude = std::min(1.0f, mag_norm.at<float>(y,x) * 5.0f);

                hsv.at<cv::Vec3f>(y,x) = cv::Vec3f(angle / 2.0f, 1.0f, magnitude);
            }
        }

        cv::Mat hsv8, bgr;
        hsv.convertTo(hsv8, CV_8UC3, 255.0);
        cv::cvtColor(hsv8, bgr, cv::COLOR_HSV2BGR);

        // blend with original frame for visualization
        cv::Mat blended;
        float alpha_blend = 0.3f;
        cv::addWeighted(frame2_bgr, 1.0f - alpha_blend, bgr, alpha_blend, 0.0, blended);

        // write heatmap frame to output video
        video_out.write(blended);

        // move to next iteration
        I1_mat = I2_mat.clone();
    }

    // cleanup
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
