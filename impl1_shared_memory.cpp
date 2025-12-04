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
#include <sys/time.h>

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
    // Shared memory declaration
    __shared__ float s_tile[18][18];  // 16x16 block + 1-pixel halo on each side

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // global coordinates subtracting 1 for halo
    int x = bx * 16 + tx - 1;
    int y = by * 16 + ty - 1;

    // Load data into shared memory
    if (x >= 0 && x < width && y >= 0 && y < height) {
        s_tile[ty][tx] = I1[y * width + x];
    }

    __syncthreads();

    // only compute internal pixels
    if (tx >= 1 && tx <= 16 && ty >= 1 && ty <= 16) {
        // check if we have a 3x3 neighborhood
        if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
            // read from shared memory
            float I_ul = s_tile[ty - 1][tx - 1];
            float I_um = s_tile[ty - 1][tx];
            float I_ur = s_tile[ty - 1][tx + 1];
            float I_ml = s_tile[ty][tx - 1];
            float I_mr = s_tile[ty][tx + 1];
            float I_ll = s_tile[ty + 1][tx - 1];
            float I_lm = s_tile[ty + 1][tx];
            float I_lr = s_tile[ty + 1][tx + 1];

            // Sobel Gx (horizontal derivative)
            // [-1  0  1
            //  -2  0  2
            //  -1  0  1]
            float gx = (-I_ul + I_ur - 2.0f * I_ml + 2.0f * I_mr - I_ll + I_lr) * 0.125f;

            // Sobel Gy (vertical derivative)
            // [-1 -2 -1
            //   0  0  0
            //   1  2  1]
            float gy = (-I_ul - 2.0f * I_um - I_ur + I_ll + 2.0f * I_lm + I_lr) * 0.125f;

            int idx = y * width + x;
            Ix[idx] = gx;
            Iy[idx] = gy;
            It[idx] = I2[idx] - I1[idx];
        }
    }
}

// Kernel to perform one iteration of Horn-Schunck update using shared memory
__global__ void horn_schunck_iteration(float* U_old, float* V_old, 
                                                   float* U_new, float* V_new,
                                                   float* Ix, float* Iy, float* It, 
                                                   int width, int height, float alpha2) {
    // Shared memory for all inputs
    __shared__ float s_U[18][18];
    __shared__ float s_V[18][18];
    __shared__ float s_Ix[18][18];
    __shared__ float s_Iy[18][18];
    __shared__ float s_It[18][18];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // global coordinates subtracting 1 for halo
    int x = bx * 16 + tx - 1;
    int y = by * 16 + ty - 1;
    
    // Load all data into shared memory
    if (x >= 0 && x < width && y >= 0 && y < height) {
        int idx = y * width + x;
        s_U[ty][tx] = U_old[idx];
        s_V[ty][tx] = V_old[idx];
        s_Ix[ty][tx] = Ix[idx];
        s_Iy[ty][tx] = Iy[idx];
        s_It[ty][tx] = It[idx];
    }
    
    __syncthreads();
    
    // only compute internal pixels
    if (tx >= 1 && tx <= 16 && ty >= 1 && ty <= 16) {
        // check bounds
        if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
            // perform division only once
            float twelveth = 1.0f / 12.0f;
            float sixth = 1.0f / 6.0f;

            // read from shared memory
            // weighted 8-neighbor average
            float U_avg = 
                (s_U[ty-1][tx-1] * twelveth +
                 s_U[ty-1][tx] * sixth +
                 s_U[ty-1][tx+1] * twelveth +
                 s_U[ty][tx-1] * sixth +
                 s_U[ty][tx+1] * sixth +
                 s_U[ty+1][tx-1] * twelveth +
                 s_U[ty+1][tx] * sixth +
                 s_U[ty+1][tx+1] * twelveth);

            float V_avg = 
                (s_V[ty-1][tx-1] * twelveth +
                 s_V[ty-1][tx] * sixth +
                 s_V[ty-1][tx+1] * twelveth +
                 s_V[ty][tx-1] * sixth +
                 s_V[ty][tx+1] * sixth +
                 s_V[ty+1][tx-1] * twelveth +
                 s_V[ty+1][tx] * sixth +
                 s_V[ty+1][tx+1] * twelveth);

            float ix_val = s_Ix[ty][tx];
            float iy_val = s_Iy[ty][tx];
            float it_val = s_It[ty][tx];
            
            float numerator = ix_val * U_avg + iy_val * V_avg + it_val;
            float denominator = alpha2 + ix_val * ix_val + iy_val * iy_val;

            int idx = y * width + x;
            U_new[idx] = U_avg - (ix_val * numerator) / denominator;
            V_new[idx] = V_avg - (iy_val * numerator) / denominator;
        }
    }
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

    // Count frames manually
    int num_frames = 0;
    cv::Mat test_frame;
    while (true) {
        cap >> test_frame;
        if (test_frame.empty()) break;
        num_frames++;
    }
    
    // Reset video to beginning
    cap.release();
    cap.open("input.mp4");

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

    dim3 block(18, 18);
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

    // timing values
    timeval start, end;

    // elapsed time vals
    double totalTime = 0.0;

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

        gettimeofday(&start, NULL);
        // Iterative refinement of flow estimates
        for (int iter = 0; iter < num_iterations; ++iter) {
            horn_schunck_iteration<<<grid, block>>>(d_U_old, d_V_old, d_U_new, d_V_new,
                                                     d_Ix, d_Iy, d_It, width, height, alpha2);
            std::swap(d_U_old, d_U_new);
            std::swap(d_V_old, d_V_new);
        }
        gettimeofday(&end, NULL);
        totalTime += (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6;

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

    printf("Total processing time for %d frames: %f seconds\n", num_frames, totalTime);

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
