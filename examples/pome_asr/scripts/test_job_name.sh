#!/bin/bash
#SBATCH --job-name=default_job       # This default will be overridden via -J if provided
#SBATCH -o logs/%x_%j.out            # %x: job name, %j: job id
#SBATCH -e logs/%x_%j.err
#SBATCH --time=00:05:00

echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"