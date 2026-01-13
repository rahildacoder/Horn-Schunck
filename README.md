To see results and a more detailed breakdown, check the pdf in the docs folder.

Rahil Badkul and Qile Jiang implement the
Horn-Schunck Method

Introduction

Goal: Estimate pixel motion between two consecutive frames in a video.

The functional E from the Horn-Schunck equation can be minimized by solving the Euler–Lagrange equations: 
where L is the integrand of E, and Δ is the Laplace operator. 
Using finite difference approximation, we get an iterative scheme: 
where u(x,y) is a weighted avg. of u around the pixel at (x,y): 

Implement H-S on GPU

We created two GPU Kernels to parallelize and optimize the calculations required for each pair of frames.

Input: Frame pair (I₁ I₂)
Step 1: Compute Intensity Derivatives (GPU kernel)
• Apply Sobel derivative operator → Ix, Iy
• Temporal difference → It
• Obtain values for each pair of frames

Step 2: Iterative Solver (GPU kernel)
• Initialize: U = 0, V = 0
• Loop num_iter: Update U, V from neighbors, Swap (U_old ↔ U_new)
Output: Flow field (U, V) 

Implement H-S on GPU

One frame
width = 1920 pixels
height = 1080 pixels

// Each block uses 18×18 threads to process a 
// 16×16 pixel region + Halo around the region

dim3 block(18, 18);    

// Divide a frame into a grid of blocks 
dim3 grid((width + 15) / 16, (height + 15) / 16);

H-S with Shared Memory
__global__ void compute_derivatives(...){

  // Step 1: Shared mem. declaration: 16x16 block + 1-pixel halo 
  __shared__ float s_tile[18][18]; 
  int x = bx * 16 + tx - 1;   // find global x coordinate 
  int y = by * 16 + ty - 1;   // find global y coordinate 

  // Step 2: Load data into shared memory
  s_tile[ty][tx] = I1[y * width + x];
   __syncthreads();       // barrier to make sure all threads finish

  // Step 3: only compute derivatives on internal pixels 
  if (tx >= 1 && tx <= 16 && ty >= 1 && ty <= 16) {
    float I_ul   = s_tile[ty - 1][tx - 1];   // read from shared mem.
    float I_um = ...
    float gx = ... 	// compute horizontal sobel derivative
    float gy = ...        // compute vertical sobel derivative
}
H = Halo (load data only)
W = Work threads  (load + compute internal pixels)
Block: shares memory

H-S with Shared Memory
__global__ void horn_schunck_iteration(...){

  // Step 1: Shared mem. declaration for 5 arrays
  __shared__ float s_U[18][18], s_V[18][18], s_Ix[18][18], ...
  
  int x = bx * 16 + tx - 1;   // find global x coordinate 
  int y = by * 16 + ty - 1;   // find global y coordinate 
  
  // Step 2: Load 5 arrays into shared memory
  s_U[ty][tx] = U_old[idx];
  s_V[ty][tx] = V_old[idx];    ...
  __syncthreads();       // barrier
  
  // Step 3: only compute on internal pixels 
  if (tx >= 1 && tx <= 16 && ty >= 1 && ty <= 16) {
    U_avg = (1/12)*corners + (1/6)*sides;    // 8-neighbor avg
    V_avg = ...
    U_new[idx] = ...;      V_new[idx] = ...
}
H = Halo (load data only)
W = Work threads  (load + compute internal pixels)
Block: shares memory

The above two kernels use shared memory. We also used distributed memory by splitting frames in a video across ranks.

H-S with Distributed Memory

Frame are distributed across MPI ranks: 
Rank 0: frames 0-16
Rank 1: frames 16-32
Rank 2: frames 32-48
Rank 3: frames 48-63
// Rank 0 loads video
if (rank == 0)
    load_frames_and_metadata();

// Rank 0 distributes frame subsets to each rank
MPI_Scatterv(flat.data(), send_counts, displs, ..., recv_buf.data(), ...);

// Each rank runs GPU optical flow on its local frames
local_max_mag = process_frame_range(local_frames, ...);

// All ranks share local max → all receive global max
MPI_Allreduce(&local_max_mag, &global_max_mag, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

// All ranks send flow results → Rank 0 collects
MPI_Gatherv(local_U.data(), ..., U_all.data(), ...);


Flow magnitudes need to be normalized, so the ranks:
Share local max
Receive global max

Roofline Analysis on HS 
44 FLOPS / pixel: 
8-point average for U: 8 mults + 7 adds = 15 FLOPs. Same for V: 15 FLOPs.
Numerator: 2 mults + 2 adds = 4 FLOPs.  Denominator: 2 + 2 = 4 FLOPs.
Update U and V: 2 × (1 mult + 1 div + 1 sub) = 6 FLOPs.  
33.3 bytes / pixel:
Block loads:     5 arrays × 18×18 tiles = 6,480 bytes.
Block writes:    2 arrays × 16×16 tiles = 2,048 bytes.  Total: 8,528 bytes per block
Per pixel:          8,528 bytes ÷ 256 pixels = 33.3 bytes/pixel

Roofline Analysis on HS tells us that the H-S kernel is memory-bound. 

Limitations:

Load balancing:

When we dealt with longer or higher resolution videos, we ran into data overflow issues
If we streamed our frames in batches and only sent new work to a rank after it finished its current batch, we could keep memory usage bounded and improve load balancing for very long / 4K videos.

Video writing:
To increase parallelization and decrease processing time further, we could offload the video writing step to the GPU.
Using a GPU-accelerated encoder would let us write frames faster and avoid the bottleneck of finishing the main computation.
