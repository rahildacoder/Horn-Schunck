#!/bin/bash
#SBATCH --job-name=horn_schunck_shared
#SBATCH --output=horn_schunck_%j.out
#SBATCH --error=horn_schunck_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=mi3001x

# Load modules 
module load rocm
module load opencv

# Compile
hipcc -O3 -o impl1_shared_memory impl1_shared_memory.cpp `pkg-config --cflags --libs opencv4`

# Run
./impl1_shared_memory
