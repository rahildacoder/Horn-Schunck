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
#include <mpi.h>

// regularization parameter and number of iterations
static float alpha = 100.0f;
static int num_iterations = 500;

// ==================== GPU KERNELS ====================

// Kernel to compute sobel image derivatives Ix, Iy, It
// Sobel derivatives in x and y directions, and temporal derivative
// This type of derivative is better for optical flow than simple finite differences
// since it smooths noise while computing gradients
__global__ void compute_derivatives(
    const float* __restrict__ I1,
    const float* __restrict__ I2,
    float* __restrict__ Ix,
    float* __restrict__ Iy,
    float* __restrict__ It,
    int width, int height)
{
    // shared tile = 16×16 work + 1 halo on each side
    __shared__ float tile[18][18];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x * 16;
    int by = blockIdx.y * 16;

    // global pixel position including halo offset
    int x = bx + tx - 1;
    int y = by + ty - 1;

    // clamp to valid image bounds
    int cx = min(max(x, 0), width - 1);
    int cy = min(max(y, 0), height - 1);

    // load tile
    tile[ty][tx] = I1[cy * width + cx];

    __syncthreads();

    // Compute only inner 16×16 region (tx=1..16, ty=1..16)
    if (tx >= 1 && tx <= 16 && ty >= 1 && ty <= 16)
    {
        // real coordinates (not clamped)
        if (x > 0 && x < width - 1 && y > 0 && y < height - 1)
        {
            float ul = tile[ty-1][tx-1];
            float um = tile[ty-1][tx];
            float ur = tile[ty-1][tx+1];
            float ml = tile[ty][tx-1];
            float mr = tile[ty][tx+1];
            float ll = tile[ty+1][tx-1];
            float lm = tile[ty+1][tx];
            float lr = tile[ty+1][tx+1];

            // Sobel X
            float gx = (-ul + ur - 2.f*ml + 2.f*mr - ll + lr);

            // Sobel Y
            float gy = (-ul - 2.f*um - ur + ll + 2.f*lm + lr);

            float it = I2[y*width + x] - I1[y*width + x];

            int idx = y * width + x;
            Ix[idx] = gx;
            Iy[idx] = gy;
            It[idx] = it;
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

// Device function to convert HSV to RGB
__device__ float3 hsv_to_rgb(float h, float s, float v) {
    // h: [0,360], s: [0,1], v: [0,1]
    float c = v * s;
    float hp = h / 60.0f;
    float x = c * (1 - fabsf(fmodf(hp, 2.0f) - 1));
    float3 rgb;

    // splitting into 6 sectors based on hue
    if (hp < 1) rgb = make_float3(c, x, 0);
    else if (hp < 2) rgb = make_float3(x, c, 0);
    else if (hp < 3) rgb = make_float3(0, c, x);
    else if (hp < 4) rgb = make_float3(0, x, c);
    else if (hp < 5) rgb = make_float3(x, 0, c);
    else rgb = make_float3(c, 0, x);

    float m = v - c;
    return make_float3(rgb.x + m, rgb.y + m, rgb.z + m);
}

// Kernel to process flow field into HSV color map
__global__ void build_flow_visualization_shared(
    const float* __restrict__ U,
    const float* __restrict__ V,
    uchar3* __restrict__ out_bgr,
    float max_mag,
    int width, int height) {
    __shared__ float s_U[18][18];
    __shared__ float s_V[18][18];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    // global pixel coords, including halo offset
    int x = bx * 16 + tx - 1;
    int y = by * 16 + ty - 1;

    // Load shared memory tile with clamping at borders
    int clamped_x = max(0, min(x, width - 1));
    int clamped_y = max(0, min(y, height - 1));

    int idx2 = clamped_y * width + clamped_x;

    s_U[ty][tx] = U[idx2];
    s_V[ty][tx] = V[idx2];


    __syncthreads();

    // Only compute interior 16×16 tile
    if (tx >= 1 && tx <= 16 && ty >= 1 && ty <= 16) {
        // check bounds
        if (x < width && y < height) {
            int idx = y * width + x;

            float u = s_U[ty][tx];
            float v = s_V[ty][tx];

            // magnitude + angle
            float mag = sqrtf(u*u + v*v);

            // normalize magnitude
            float norm_mag = mag / (max_mag + 1e-6f);

            // apply contrast enhancement
            float m = sqrtf(fminf(norm_mag, 1.0f));

            // angle in degrees
            float ang = atan2f(v, u) * 180.0f / M_PI;
            if (ang < 0) ang += 360.0f;

            // shift angle by 180 degrees for better visualization
            ang = fmodf(ang + 180.0f, 360.0f);

            // convert HSV to RGB
            float3 rgb = hsv_to_rgb(ang, 1.0, m);

            // convert to uchar BGR (OpenCV format)
            out_bgr[idx] = make_uchar3(
                (unsigned char)(rgb.z * 255.0f),
                (unsigned char)(rgb.y * 255.0f),
                (unsigned char)(rgb.x * 255.0f)
            );
        }
    }
}

// Kernel to blend frames
__global__ void blend_frames_kernel(
    const uchar3* __restrict__ frame2,
    const uchar3* __restrict__ flow_vis,
    uchar3* __restrict__ out,
    float a, float b,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    uchar3 f2 = frame2[idx];
    uchar3 fv = flow_vis[idx];

    // Weighted blending
    uchar3 result;
    result.x = (unsigned char)(a * f2.x + b * fv.x);
    result.y = (unsigned char)(a * f2.y + b * fv.y);
    result.z = (unsigned char)(a * f2.z + b * fv.z);

    out[idx] = result;
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

// Reads all frames from input.mp4 into all_frames vector for rank 0
void load_frames_and_metadata(std::vector<cv::Mat> &all_frames, int &width, int &height, float &fps, int &num_frames) {
    cv::VideoCapture cap("input.mp4");
    if (!cap.isOpened())
    {
        std::cerr << "ERROR: Cannot open input.mp4\n";
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    fps = (float)cap.get(cv::CAP_PROP_FPS);

    cv::Mat frame;
    while (true)
    {
        cap >> frame;
        if (frame.empty()) break;
        all_frames.push_back(frame.clone());
    }

    num_frames = (int)all_frames.size();

    std::cout << "Rank 0 loaded " << num_frames
              << " frames (" << width << "x" << height
              << " @ " << fps << " fps)\n";
}

// compute send_counts and displacements for scattering frames
void compute_frame_distribution(int world, int num_pairs,
                                int width, int height,
                                std::vector<int> &send_counts,
                                std::vector<int> &displs)
{
    send_counts.assign(world, 0);
    displs.assign(world, 0);

    int pixels3 = width * height * 3;
    int base  = num_pairs / world;
    int extra = num_pairs % world;

    int offset_bytes = 0;

    for (int r = 0; r < world; r++)
    {
        int r_pairs  = base + (r < extra ? 1 : 0);
        int r_frames = (r_pairs > 0) ? (r_pairs + 1) : 0;

        int r_start_pair  = r * base + std::min(r, extra);
        int r_start_frame = r_start_pair;

        send_counts[r] = r_frames * pixels3;
        displs[r]      = r_start_frame * pixels3;

        offset_bytes += send_counts[r];
    }
}

void scatter_frames(const std::vector<cv::Mat> &all_frames,
                    std::vector<cv::Mat> &local_frames,
                    const std::vector<int> &send_counts,
                    const std::vector<int> &displs,
                    int rank, int world,
                    int width, int height,
                    int num_frames)
{
    int frame_bytes = width * height * 3;

    // Flatten frames on rank 0
    std::vector<uchar> flat;
    if (rank == 0)
    {
        flat.resize(num_frames * frame_bytes);
        for (int f = 0; f < num_frames; f++)
        {
            memcpy(flat.data() + f * frame_bytes,
                   all_frames[f].data,
                   frame_bytes);
        }
    }

    // Broadcast counts/displs
    MPI_Bcast((void*)send_counts.data(), world, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast((void*)displs.data(),      world, MPI_INT, 0, MPI_COMM_WORLD);

    // Receive buffer
    std::vector<uchar> recv_buf(send_counts[rank]);

    // Scatter actual bytes
    MPI_Scatterv(rank == 0 ? flat.data() : nullptr,
                 (int*)send_counts.data(), (int*)displs.data(), MPI_UNSIGNED_CHAR,
                 recv_buf.data(), send_counts[rank], MPI_UNSIGNED_CHAR,
                 0, MPI_COMM_WORLD);

    // Rebuild cv::Mat frames
    int local_frames_count = send_counts[rank] / frame_bytes;
    local_frames.resize(local_frames_count);

    for (int i = 0; i < local_frames_count; i++)
    {
        uchar *ptr = recv_buf.data() + i * frame_bytes;
        local_frames[i] = cv::Mat(height, width, CV_8UC3, ptr).clone();
    }
}

void gather_flows(const std::vector<float> &local_U,
                  const std::vector<float> &local_V,
                  std::vector<float> &U_all,
                  std::vector<float> &V_all,
                  int local_pairs,
                  int rank, int world,
                  int num_pairs, int pixels)
{
    std::vector<int> flow_counts(world), flow_displs(world);

    if (rank == 0)
    {
        int base  = num_pairs / world;
        int extra = num_pairs % world;

        int offset = 0;
        for (int r = 0; r < world; r++)
        {
            int r_pairs = base + (r < extra ? 1 : 0);
            int elems   = r_pairs * pixels;

            flow_counts[r] = elems;
            flow_displs[r] = offset;
            offset += elems;
        }

        U_all.resize(num_pairs * pixels);
        V_all.resize(num_pairs * pixels);
    }

    int local_elems = local_pairs * pixels;

    MPI_Gatherv(local_U.data(), local_elems, MPI_FLOAT,
                rank == 0 ? U_all.data() : nullptr,
                flow_counts.data(), flow_displs.data(), MPI_FLOAT,
                0, MPI_COMM_WORLD);

    MPI_Gatherv(local_V.data(), local_elems, MPI_FLOAT,
                rank == 0 ? V_all.data() : nullptr,
                flow_counts.data(), flow_displs.data(), MPI_FLOAT,
                0, MPI_COMM_WORLD);
}

void visualize_and_write_video(const std::vector<cv::Mat> &all_frames,
                               std::vector<float> &U_all,
                               std::vector<float> &V_all,
                               int width, int height,
                               int num_pairs, float fps,
                               float global_max_mag)
{
    const int pixels    = width * height;
    const int size_f    = pixels * sizeof(float);
    const int size_uc3  = pixels * sizeof(uchar3);

    cv::VideoWriter writer(
        "optical_flow_output.mp4",
        cv::VideoWriter::fourcc('a','v','c','1'),
        fps,
        cv::Size(width, height)
    );

    if (!writer.isOpened())
    {
        std::cerr << "ERROR: cannot open VideoWriter.\n";
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    // GPU allocate for visualization
    float  *d_U, *d_V;
    uchar3 *d_bgr, *d_frame, *d_blended;

    hipMalloc(&d_U, size_f);
    hipMalloc(&d_V, size_f);
    hipMalloc(&d_bgr, size_uc3);
    hipMalloc(&d_frame, size_uc3);
    hipMalloc(&d_blended, size_uc3);

    dim3 block(18, 18);
    dim3 grid ((width + 15)/16, (height + 15)/16);

    dim3 blockB(16,16);
    dim3 gridB ((width + 15)/16, (height + 15)/16);

    std::vector<uchar3> h_blended(pixels);

    for (int p = 0; p < num_pairs; p++)
    {
        float *U_ptr = U_all.data() + p * pixels;
        float *V_ptr = V_all.data() + p * pixels;

        // Apply magnitude threshold to remove noise before visualization
        float epsilon = 0.05f;
        for (int i = 0; i < pixels; i++) {
            float m = std::sqrt(U_ptr[i]*U_ptr[i] + V_ptr[i]*V_ptr[i]);
            if (m < epsilon) {
                U_ptr[i] = 0.0f;
                V_ptr[i] = 0.0f;
            }
        }

        hipMemcpy(d_U, U_ptr, size_f, hipMemcpyHostToDevice);
        hipMemcpy(d_V, V_ptr, size_f, hipMemcpyHostToDevice);

        build_flow_visualization_shared<<<grid, block>>>(
            d_U, d_V, d_bgr, global_max_mag, width, height
        );

        hipMemcpy(d_frame,
                  all_frames[p+1].ptr<uchar3>(),
                  size_uc3,
                  hipMemcpyHostToDevice);

        blend_frames_kernel<<<gridB, blockB>>>(
            d_frame, d_bgr, d_blended,
            0.70f, 0.30f,
            width, height
        );

        hipMemcpy(h_blended.data(), d_blended, size_uc3, hipMemcpyDeviceToHost);

        writer.write(cv::Mat(height, width, CV_8UC3, h_blended.data()));
    }

    hipFree(d_U);
    hipFree(d_V);
    hipFree(d_bgr);
    hipFree(d_frame);
    hipFree(d_blended);

    writer.release();

    std::cout << "Rank 0 wrote " << num_pairs
              << " frames with global_max_mag=" << global_max_mag << "\n";
}

// Process video frames in range for each rank of MPI
// frames: length = num_pairs + 1 (or 0 if num_pairs == 0)
// Output: U_all, V_all packed as [pair][pixel], and return local max magnitude
float process_frame_range(const std::vector<cv::Mat>& frames, int num_pairs, int width,
    int height, std::vector<float>& U_all, std::vector<float>& V_all, int rank) {
    // This function processes frame[start] ... frame[end]
    // using GPU Horn–Schunck

    const int pixels = width * height;
    const int size_f = pixels * sizeof(float);
    const int size_uc3 = pixels * sizeof(uchar3);

    // Allocate packed output
    U_all.assign(num_pairs * pixels, 0.0f);
    V_all.assign(num_pairs * pixels, 0.0f);

    // GPU memory allocation
    float *d_I1, *d_I2, *d_Ix, *d_Iy, *d_It;
    float *d_U_old, *d_U_new, *d_V_old, *d_V_new;

    hipMalloc(&d_I1, size_f);
    hipMalloc(&d_I2, size_f);
    hipMalloc(&d_Ix, size_f);
    hipMalloc(&d_Iy, size_f);
    hipMalloc(&d_It, size_f);
    hipMalloc(&d_U_old, size_f);
    hipMalloc(&d_U_new, size_f);
    hipMalloc(&d_V_old, size_f);
    hipMalloc(&d_V_new, size_f);

    // Define block and grid sizes for kernels
    dim3 block(18, 18);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    float alpha2 = alpha * alpha;

    // Host buffers for a single frame's flow
    std::vector<float> U_buf(pixels);
    std::vector<float> V_buf(pixels);

    // Track local max across all frames processed by this rank
    float local_max_mag = 0.0f;

    // Timing setup (do once before loop over pairs)
    static bool first_time = true;
    static hipEvent_t start, stop;
    static float total_kernel_time = 0.0f;
    static int total_iterations = 0;

    if (first_time) {
        hipEventCreate(&start);
        hipEventCreate(&stop);
        first_time = false;
    }

    // Convert first frame in range to grayscale float
    cv::Mat I1, I2;
    cv::cvtColor(frames[0], I1, cv::COLOR_BGR2GRAY);
    I1.convertTo(I1, CV_32F, 1.0 / 255.0);

    // Loop over frame pairs inside the assigned range
    for (int i = 0; i < num_pairs; i++) {
        // Prepare next frame
        cv::cvtColor(frames[i+1], I2, cv::COLOR_BGR2GRAY);
        I2.convertTo(I2, CV_32F, 1.0 / 255.0);

        // Copy grayscale frames to GPU
        hipMemcpy(d_I1, I1.ptr<float>(), size_f, hipMemcpyHostToDevice);
        hipMemcpy(d_I2, I2.ptr<float>(), size_f, hipMemcpyHostToDevice);

        // Compute Sobel derivatives
        compute_derivatives<<<grid, block>>>(d_I1, d_I2, d_Ix, d_Iy, d_It, width, height);

        // Initialize flow fields
        hipMemset(d_U_old, 0, size_f);
        hipMemset(d_V_old, 0, size_f);

        // Iterative Horn–Schunck solver with timing
        hipEventRecord(start);
        for (int iter = 0; iter < num_iterations; iter++) {
            horn_schunck_iteration<<<grid, block>>>(d_U_old, d_V_old, d_U_new, d_V_new, d_Ix, d_Iy, d_It, width, height, alpha2);
            std::swap(d_U_old, d_U_new);
            std::swap(d_V_old, d_V_new);
        }
        hipEventRecord(stop);
        hipEventSynchronize(stop);

        float milliseconds = 0;
        hipEventElapsedTime(&milliseconds, start, stop);
        total_kernel_time += milliseconds;
        total_iterations += num_iterations;

        // Copy results to CPU for this pair
        hipMemcpy(U_buf.data(), d_U_old, size_f, hipMemcpyDeviceToHost);
        hipMemcpy(V_buf.data(), d_V_old, size_f, hipMemcpyDeviceToHost);

        // Store into packed arrays and update local max magnitude
        int base_idx = i * pixels;
        for (int k = 0; k < pixels; k++) {
            float u = U_buf[k];
            float v = V_buf[k];
            float m = std::sqrt(u * u + v * v);

            U_all[base_idx + k] = u;
            V_all[base_idx + k] = v;

            if (m > local_max_mag) local_max_mag = m;
        }

        // Move to next iteration
        I1 = I2.clone();
    }

    // Print timing results
    if (num_pairs > 0) {
        float avg_time_per_iter = total_kernel_time / total_iterations;
        long long total_flops = 44LL * pixels * total_iterations;
        float achieved_gflops = (total_flops / 1e9) / (total_kernel_time / 1000.0);

        printf("Rank %d: Avg kernel time per iteration: %.3f ms\n", rank, avg_time_per_iter);
        printf("Rank %d: Total kernel time: %.3f ms\n", rank, total_kernel_time);
        printf("Rank %d: Total iterations: %d\n", rank, total_iterations);
        printf("Rank %d: Achieved performance: %.1f GFLOP/s\n", rank, achieved_gflops);
    }

    // Cleanup GPU memory
    hipFree(d_I1);
    hipFree(d_I2);
    hipFree(d_Ix);
    hipFree(d_Iy);
    hipFree(d_It);
    hipFree(d_U_old);
    hipFree(d_U_new);
    hipFree(d_V_old);
    hipFree(d_V_new);

    return local_max_mag;
}

// ==================== MAIN ====================

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, world;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    int width=0, height=0, num_frames=0;
    float fps=0.0f;
    std::vector<cv::Mat> all_frames;

    // Rank 0 loads video
    if (rank == 0)
        load_frames_and_metadata(all_frames, width, height, fps, num_frames);

    // Broadcast metadata
    MPI_Bcast(&width,       1, MPI_INT,   0, MPI_COMM_WORLD);
    MPI_Bcast(&height,      1, MPI_INT,   0, MPI_COMM_WORLD);
    MPI_Bcast(&fps,         1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_frames,  1, MPI_INT,   0, MPI_COMM_WORLD);

    int num_pairs = (num_frames > 0 ? num_frames - 1 : 0);
    int pixels    = width * height;

    // Compute distribution
    std::vector<int> send_counts, displs;
    compute_frame_distribution(world, num_pairs, width, height,
                               send_counts, displs);

    // Scatter the frames
    std::vector<cv::Mat> local_frames;
    scatter_frames(all_frames, local_frames,
                   send_counts, displs,
                   rank, world,
                   width, height, num_frames);

    // Determine how many pairs this rank owns
    int local_frames_count = (int)local_frames.size();
    int local_pairs        = (local_frames_count > 0 ? local_frames_count - 1 : 0);

    // Compute local flow fields
    std::vector<float> local_U, local_V;
    float local_max_mag = 0.0f;

    if (local_pairs > 0)
        local_max_mag = process_frame_range(
            local_frames, local_pairs,
            width, height,
            local_U, local_V,
            rank
        );

    // Find global max magnitude
    float global_max_mag = 0.0f;
    MPI_Allreduce(&local_max_mag, &global_max_mag,
                  1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

    // Gather flows
    std::vector<float> U_all, V_all;
    gather_flows(local_U, local_V,
                 U_all, V_all,
                 local_pairs,
                 rank, world,
                 num_pairs, pixels);

    // Rank 0 writes output
    if (rank == 0)
        visualize_and_write_video(all_frames, U_all, V_all,
                                  width, height,
                                  num_pairs, fps,
                                  global_max_mag);

    MPI_Finalize();
    return 0;
}
