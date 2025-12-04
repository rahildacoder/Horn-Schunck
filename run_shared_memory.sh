#!/bin/bash
#SBATCH -J horn_schunck_shared
#SBATCH -o horn_schunck.log
#SBATCH -t 00:10:00
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p mi3001x

# Load rocm
module load rocm

# Activate conda environment with OpenCV
source $HOME/miniconda3/bin/activate horn
conda activate horn

# Compile with conda's OpenCV
hipcc -O3 -o impl1_shared_memory impl1_shared_memory.cpp \
    -I$CONDA_PREFIX/include/opencv4 \
    -L$CONDA_PREFIX/lib \
    -lopencv_core -lopencv_imgproc -lopencv_videoio -lopencv_imgcodecs

# Set library path for runtime
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Run
./impl1_shared_memory
