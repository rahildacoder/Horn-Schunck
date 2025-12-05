#!/bin/bash
#SBATCH -J roofline_multiresolution
#SBATCH -o roofline_all.log
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p mi3001x

module load rocm
module load openmpi4/4.1.8
source $HOME/miniconda3/bin/activate horn
conda activate horn

# Compile
hipcc --offload-arch=gfx942 -O3 impl1_shared_memory_timed.cpp \
    -I$CONDA_PREFIX/include/opencv4 \
    -I/opt/ohpc/pub/mpi/openmpi4-gnu12/4.1.8/include \
    -L$CONDA_PREFIX/lib \
    -L/opt/ohpc/pub/mpi/openmpi4-gnu12/4.1.8/lib \
    -L/opt/rocm-7.1.0/lib \
    -lopencv_core -lopencv_imgproc -lopencv_videoio -lopencv_imgcodecs \
    -lamdhip64 -lmpi \
    -o impl_mpi_timed

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
mkdir -p results

# Resolutions and number of runs
resolutions=("360p" "720p" "1080p" "1440p" "2160p")
num_runs=3

# Run experiments
for res in "${resolutions[@]}"; do
    echo "=== Testing $res ==="

    [ ! -f "input_${res}.mp4" ] && echo "Missing input_${res}.mp4" && continue

    cp "input_${res}.mp4" input.mp4

    # Multiple runs for averaging
    for run in $(seq 1 $num_runs); do
        echo "Run $run/$num_runs:"
        srun -n 4 ./impl_mpi_timed 2>&1 | tee "results/log_${res}_run${run}.txt"
        [ -f "optical_flow_output.mp4" ] && rm optical_flow_output.mp4
    done

    # Extract and average GFLOP/s
    avg_gflops=$(grep "Achieved performance" results/log_${res}_run*.txt | \
                 awk '{sum+=$(NF-1); count++} END {printf "%.1f", sum/count}')

    echo "$res: Average GFLOP/s = $avg_gflops"
    echo ""
done

echo "=== Summary ==="
for res in "${resolutions[@]}"; do
    avg=$(grep "Achieved performance" results/log_${res}_run*.txt 2>/dev/null | \
          awk '{sum+=$(NF-1); count++} END {printf "%.1f", sum/count}')
    [ -n "$avg" ] && echo "$res: $avg GFLOP/s"
done
