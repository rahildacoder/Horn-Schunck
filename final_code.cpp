#include <hip/hip_runtime.h>
#include <math.h>
#include <stdio.h>
#include <sys/time.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <mpi.h>

// regularization parameter and number of iterations
static float alpha = 5.0f;
static int num_iterations = 20;

// ==================== GPU KERNELS ====================

// Kernel to compute sobel image derivatives Ix, Iy, It
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
            float gx = (-I_ul + I_ur - 2.0f * I_ml + 2.0f * I_mr - I_ll + I_lr) * 0.125f;

            // Sobel Gy (vertical derivative)
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
            float twelveth = 1.0f / 12.0f;
            float sixth    = 1.0f / 6.0f;

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

// ==================== CPU HELPERS ====================

// get the width, height, fps of a video file
void get_video_properties(const char* filename, int& width, int& height, float& fps) {
    cv::VideoCapture cap(filename);
    if (!cap.isOpened()) {
        printf("Error opening video file %s\n", filename);
        width = height = 0;
        fps = 0.0f;
        return;
    }
    width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    fps = static_cast<float>(cap.get(cv::CAP_PROP_FPS));
    cap.release();
}

// ==================== MAIN (MPI + GPU) ====================

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    int width = 0, height = 0;
    float fps = 0.0f;

    get_video_properties("input_360p.mp4", width, height, fps);

    if (width == 0 || height == 0 || fps <= 0.0f) {
        if (rank == 0) {
            printf("Error: could not read video properties from input_360p.mp4\n");
        }
        MPI_Finalize();
        return -1;
    }

    // Rank 0 counts total frames
    int num_frames = 0;
    if (rank == 0) {
        cv::VideoCapture cap0("input_360p.mp4");
        if (!cap0.isOpened()) {
            printf("Rank 0: Error opening input_360p.mp4\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        cv::Mat tmp;
        while (true) {
            cap0 >> tmp;
            if (tmp.empty()) break;
            num_frames++;
        }
        cap0.release();

        printf("Rank 0: Video properties - Frames: %d, Width: %d, Height: %d, FPS: %.2f\n",
               num_frames, width, height, fps);
    }

    // Broadcast metadata to all ranks
    MPI_Bcast(&num_frames, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&fps, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (num_frames < 2) {
        if (rank == 0) {
            printf("Not enough frames in video.\n");
        }
        MPI_Finalize();
        return 0;
    }

    int num_pairs = num_frames - 1;

    // Time-domain decomposition over frame pairs
    int base  = num_pairs / world;         // minimum pairs per rank
    int extra = num_pairs % world;         // first 'extra' ranks get +1 pair

    int local_pairs = base + (rank < extra ? 1 : 0);
    int start_pair = rank * base + std::min(rank, extra);

    if (local_pairs <= 0) {
        // This rank has no work
        MPI_Finalize();
        return 0;
    }

    // Each rank opens the video independently
    cv::VideoCapture cap("input_360p.mp4");
    if (!cap.isOpened()) {
        printf("Rank %d: Error opening input_360p.mp4\n", rank);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    // Skip frames up to start_pair
    cv::Mat dummy, frame1_bgr;
    for (int i = 0; i < start_pair; ++i) {
        cap >> dummy;
        if (dummy.empty()) {
            printf("Rank %d: Ran out of frames while skipping to start_pair=%d\n",
                   rank, start_pair);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }

    // Read the first frame this rank will use
    cap >> frame1_bgr;
    if (frame1_bgr.empty()) {
        printf("Rank %d: Could not read initial frame at pair %d\n", rank, start_pair);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    // GPU memory allocation
    int n_pixels = width * height;
    int size = n_pixels * sizeof(float);

    float *d_I1, *d_I2, *d_Ix, *d_Iy, *d_It;
    float *d_U_old, *d_U_new, *d_V_old, *d_V_new;
    hipMalloc(&d_I1, size);
    hipMalloc(&d_I2, size);
    hipMalloc(&d_Ix, size);
    hipMalloc(&d_Iy, size);
    hipMalloc(&d_It,size);
    hipMalloc(&d_U_old, size);
    hipMalloc(&d_U_new, size);
    hipMalloc(&d_V_old, size);
    hipMalloc(&d_V_new, size);

    dim3 block(18, 18);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    float alpha2 = alpha * alpha;

    // Host arrays for result
    std::vector<float> U(n_pixels);
    std::vector<float> V(n_pixels);

    cv::Mat I1_mat, I2_mat;
    cv::Mat mag(height, width, CV_32F);
    cv::Mat ang(height, width, CV_32F);

    // Convert first frame to grayscale float
    cv::cvtColor(frame1_bgr, I1_mat, cv::COLOR_BGR2GRAY);
    I1_mat.convertTo(I1_mat, CV_32F, 1.0 / 255.0);

    // Store processed frames locally on each rank
    std::vector<cv::Mat> processed_blended;
    std::vector<cv::Mat> processed_overlay;
    processed_blended.reserve(local_pairs);
    processed_overlay.reserve(local_pairs);

    // timing for Horn-Schunck iterations
    timeval start, end;
    double localTime = 0.0;
    int processed_pairs = 0;

    // ---- Process this rank's subset of frame pairs ----
    for (int p = 0; p < local_pairs; ++p) {
        cv::Mat frame2_bgr;
        cap >> frame2_bgr;
        if (frame2_bgr.empty()) {
            printf("Rank %d: Ran out of frames while processing local pair %d (global pair %d)\n",
                   rank, p, start_pair + p);
            break;
        }

        cv::cvtColor(frame2_bgr, I2_mat, cv::COLOR_BGR2GRAY);
        I2_mat.convertTo(I2_mat, CV_32F, 1.0 / 255.0);

        // Copy frames to GPU
        hipMemcpy(d_I1, I1_mat.ptr<float>(), size, hipMemcpyHostToDevice);
        hipMemcpy(d_I2, I2_mat.ptr<float>(), size, hipMemcpyHostToDevice);

        // Compute derivatives
        compute_derivatives<<<grid, block>>>(d_I1, d_I2, d_Ix, d_Iy, d_It, width, height);

        // Initialize flow
        hipMemset(d_U_old, 0, size);
        hipMemset(d_V_old, 0, size);

        gettimeofday(&start, NULL);
        for (int iter = 0; iter < num_iterations; ++iter) {
            horn_schunck_iteration<<<grid, block>>>(
                d_U_old, d_V_old, d_U_new, d_V_new,
                d_Ix, d_Iy, d_It, width, height, alpha2
            );
            std::swap(d_U_old, d_U_new);
            std::swap(d_V_old, d_V_new);
        }
        hipDeviceSynchronize();
        gettimeofday(&end, NULL);
        localTime += (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6;

        // Copy result back
        hipMemcpy(U.data(), d_U_old, size, hipMemcpyDeviceToHost);
        hipMemcpy(V.data(), d_V_old, size, hipMemcpyDeviceToHost);

        // Build mag and ang matrices
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = y * width + x;
                float u = U[idx];
                float v = V[idx];
                mag.at<float>(y, x) = sqrtf(u*u + v*v);
                ang.at<float>(y, x) = atan2f(v, u);
            }
        }

        // Normalize magnitudes to [0,1]
        cv::Mat mag_norm;
        cv::normalize(mag, mag_norm, 0.0f, 1.0f, cv::NORM_MINMAX);

        cv::Mat hsv(height, width, CV_32FC3);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float angle = ang.at<float>(y, x) * 180.0f / M_PI;
                if (angle < 0) angle += 360.0f;

                float magnitude = std::min(1.0f, mag_norm.at<float>(y, x) * 5.0f);
                hsv.at<cv::Vec3f>(y, x) = cv::Vec3f(angle / 2.0f, 1.0f, magnitude);
            }
        }

        cv::Mat hsv8, bgr_overlay;
        hsv.convertTo(hsv8, CV_8UC3, 255.0);
        cv::cvtColor(hsv8, bgr_overlay, cv::COLOR_HSV2BGR);

        // blend with original frame
        cv::Mat blended;
        float alpha_blend = 0.3f;
        cv::addWeighted(frame1_bgr, 1.0f - alpha_blend, bgr_overlay, alpha_blend, 0.0, blended);

        // Store both overlay-only and blended frames
        processed_overlay.push_back(bgr_overlay.clone());
        processed_blended.push_back(blended.clone());
        processed_pairs++;

        // Next iteration: current frame becomes previous
        frame1_bgr = frame2_bgr.clone();
        I1_mat = I2_mat.clone();
    }

    // Free GPU memory on each rank
    hipFree(d_I1);
    hipFree(d_I2);
    hipFree(d_Ix);
    hipFree(d_Iy);
    hipFree(d_It);
    hipFree(d_U_old);
    hipFree(d_U_new);
    hipFree(d_V_old);
    hipFree(d_V_new);

    cap.release();

    // MPI Gatherv to combine processed frames on rank 0

    int frame_bytes = width * height * 3; // uchar BGR
    int local_frames = processed_pairs;
    int local_bytes = local_frames * frame_bytes;

    // Serialize local frames into byte buffers (blended + overlay)
    std::vector<uchar> sendbuf_blended(local_bytes);
    std::vector<uchar> sendbuf_overlay(local_bytes);

    for (int i = 0; i < local_frames; i++) {
        memcpy(sendbuf_blended.data() + i * frame_bytes,
               processed_blended[i].ptr<uchar>(),
               frame_bytes);

        memcpy(sendbuf_overlay.data() + i * frame_bytes,
               processed_overlay[i].ptr<uchar>(),
               frame_bytes);
    }

    // Rank 0 will receive sizes and data
    std::vector<int> recv_counts;
    std::vector<int> displs;
    if (rank == 0) {
        recv_counts.resize(world);
    }

    // Gather byte counts from all ranks (same for blended and overlay)
    MPI_Gather(&local_bytes, 1, MPI_INT,
               rank == 0 ? recv_counts.data() : nullptr, 1, MPI_INT,
               0, MPI_COMM_WORLD);

    std::vector<uchar> recvbuf_blended;
    std::vector<uchar> recvbuf_overlay;

    // Prepare displacements and total receive buffer sizes on rank 0
    if (rank == 0) {
        displs.resize(world);
        int total_bytes = 0;
        for (int r = 0; r < world; r++) {
            displs[r] = total_bytes;
            total_bytes += recv_counts[r];
        }
        recvbuf_blended.resize(total_bytes);
        recvbuf_overlay.resize(total_bytes);
    }

    // Gather the actual blended frame data
    MPI_Gatherv(sendbuf_blended.data(), local_bytes, MPI_UNSIGNED_CHAR,
                rank == 0 ? recvbuf_blended.data() : nullptr,
                rank == 0 ? recv_counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);

    // Gather the actual overlay-only frame data
    MPI_Gatherv(sendbuf_overlay.data(), local_bytes, MPI_UNSIGNED_CHAR,
                rank == 0 ? recvbuf_overlay.data() : nullptr,
                rank == 0 ? recv_counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);

    // Rank 0 reconstructs frames in rank order and writes two videos
    if (rank == 0) {
        cv::VideoWriter writer_blended(
            "optical_flow_blended.mp4",
            cv::VideoWriter::fourcc('a','v','c','1'),
            fps,
            cv::Size(width, height)
        );

        cv::VideoWriter writer_overlay(
            "optical_flow_overlay.mp4",
            cv::VideoWriter::fourcc('a','v','c','1'),
            fps,
            cv::Size(width, height)
        );

        if (!writer_blended.isOpened() || !writer_overlay.isOpened()) {
            printf("Rank 0: ERROR: could not open output writers\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        // Reconstruct frames in rank order
        int offset = 0;
        for (int r = 0; r < world; r++) {
            int bytes_r = recv_counts[r];
            int frames_r = (bytes_r / frame_bytes);

            for (int i = 0; i < frames_r; i++) {
                uchar* ptr_b = recvbuf_blended.data() + offset + i * frame_bytes;
                uchar* ptr_o = recvbuf_overlay.data() + offset + i * frame_bytes;

                cv::Mat frame_b(height, width, CV_8UC3, ptr_b);
                cv::Mat frame_o(height, width, CV_8UC3, ptr_o);

                writer_blended.write(frame_b.clone());
                writer_overlay.write(frame_o.clone());
            }
            offset += bytes_r;
        }

        // Release writers
        writer_blended.release();
        writer_overlay.release();
    }

    // Gather and print total processing time
    double totalTimeGlobal = 0.0;
    MPI_Reduce(&localTime, &totalTimeGlobal, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Total processing time over all ranks (sum of GPU iteration times): %f seconds\n",
               totalTimeGlobal);
    }

    MPI_Finalize();

    return 0;
}