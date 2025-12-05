#!/bin/bash
#SBATCH -J horn_schunck_timed
#SBATCH -o horn_schunck_timed.log
#SBATCH -t 00:10:00
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p mi3001x

# Load rocm
module load rocm
module load openmpi4/4.1.8

# Activate conda environment with OpenCV
source $HOME/miniconda3/bin/activate horn
conda activate horn

# Compile with conda's OpenCV
hipcc --offload-arch=gfx942 -O3 impl1_shared_memory_timed.cpp \
    -I$CONDA_PREFIX/include/opencv4 \
    -I/opt/ohpc/pub/mpi/openmpi4-gnu12/4.1.8/include \
    -L$CONDA_PREFIX/lib \
    -L/opt/ohpc/pub/mpi/openmpi4-gnu12/4.1.8/lib \
    -L/opt/rocm-7.1.0/lib \
    -lopencv_core -lopencv_imgproc -lopencv_videoio -lopencv_imgcodecs \
    -lamdhip64 \
    -lmpi \
    -o impl_mpi_timed

# Set library path for runtime
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Run MPI implementation with timing
srun -n 4 ./impl_mpi_timed
