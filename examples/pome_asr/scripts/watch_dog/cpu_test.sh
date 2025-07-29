#!/bin/bash -l
#SBATCH -N 1                   # Use 1 node
#SBATCH -c 12                  # Use 12 CPU cores
#SBATCH -p cpu                 # Change the partition to a CPU partition (if available)
#SBATCH --time=08:00:00         # Set the job time limit
#SBATCH --output=log/%j.output  # Standard output
#SBATCH --error=log/%j.error    # Standard error
#SBATCH --mail-type=BEGIN,END,FAIL  # Notifications on job start, end, or fail
#SBATCH --mail-user=jindaznb@gmail.com  # Email notifications

# Remove GPU-specific settings for CPU testing
# export CUDA_VISIBLE_DEVICES=0   # This line should be commented out or removed for CPU usage
# export TOKENIZERS_PARALLELISM=false # Not needed for CPU, but can leave it if you want
# export CUDA_LAUNCH_BLOCKING=1    # Not needed for CPU, remove or comment out
export OMP_NUM_THREADS=12         # Use 12 CPU threads (adjust based on the number of cores)
export HYDRA_FULL_ERROR=1         # Enable full error reporting for Hydra

# Load necessary modules for CPU testing
module purge                    # Clear previously loaded modules
module load discovery            # Load the discovery module
module load python/3.8.1         # Load Python 3.8.1
module load anaconda3/3.7        # Load Anaconda 3.7
module load ffmpeg/20190305      # Load FFmpeg (if needed)

# Activate the virtual environment
source activate /work/van-speech-nlp/jindaznb/slamenv/

# Your testing command goes here
# Example: running a Python script that tests the environment
# python test_environment.py
