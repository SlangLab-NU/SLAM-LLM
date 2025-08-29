#!/bin/bash

# Create logs directory
mkdir -p logs

# Background build and run script with comprehensive logging
exec > >(tee -a logs/build_and_run_$(date +%Y%m%d_%H%M%S).log) 2>&1

echo "Starting Singularity container build at $(date)"
echo "=========================================="

# Build container
singularity build slam_llm.sif slam_llm.def

if [ $? -eq 0 ]; then
    echo "Container build completed successfully at $(date)"
    echo "Container size: $(ls -lh slam_llm.sif)"
    
    # Test container
    echo "Testing container..."
    singularity exec slam_llm.sif python --version
    singularity exec slam_llm.sif python -c "import torch; print(f'PyTorch: {torch.__version__}')"
    
    echo "Submitting SLURM job..."
    cd examples/asr_librispeech/scripts
    sbatch 00_dev_test_run1_singularity.sh
    
    echo "Job submitted at $(date). Check queue with 'squeue -u $(whoami)'"
    
else
    echo "Container build failed at $(date)"
    exit 1
fi

echo "Build and submission completed at $(date)"