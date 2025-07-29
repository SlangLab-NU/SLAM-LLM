#!/bin/bash -l
#SBATCH -N 1
#SBATCH -c 12
#SBATCH -p gpu
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=log/%j.output
#SBATCH --error=log/%j.error

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1
export HYDRA_FULL_ERROR=1

module purge
module load discovery
module load python/3.8.1 
module load anaconda3/3.7 
module load ffmpeg/20190305 
source activate /work/van-speech-nlp/jindaznb/slamenv/
which python

# Generate a timestamp for the log file
timestamp=$(date +"%Y%m%d_%H%M%S")

# Create the terminal_log folder if it doesn't exist
mkdir -p terminal_log

# Redirect output to the log file in terminal_log folder
exec > >(tee -a "terminal_log/slam_run_${timestamp}.txt") 2>&1


bash train_eval.sh \
    --llm_name llama32_1b \
    --task test \
    --encoder_config wavlm-mono \
    --num_epochs 10 \
    --batch_size_training 4 \
    --use_peft true \
    --test_run \
    --save_embedding true \
    --train_data_folder psst_phoneme \


bash train_eval.sh \
    --llm_name llama32_1b \
    --task test \
    --encoder_config wavlm-mono \
    --num_epochs 10 \
    --batch_size_training 4 \
    --use_peft true \
    --test_run \
    --save_embedding true \
    --train_data_folder ami \



bash train_eval.sh \
    --llm_name llama32_1b \
    --task test \
    --encoder_config wavlm-mono \
    --num_epochs 10 \
    --batch_size_training 4 \
    --use_peft true \
    --test_run \
    --save_embedding true \
    --train_data_folder librispeech-100 \


bash train_eval.sh \
    --llm_name llama32_1b \
    --task test \
    --encoder_config wavlm-mono \
    --num_epochs 10 \
    --batch_size_training 4 \
    --use_peft true \
    --test_run \
    --save_embedding true \
    --train_data_folder librispeech-100_phoneme \


bash train_eval.sh \
    --llm_name llama32_1b \
    --task test \
    --encoder_config wavlm-mono \
    --num_epochs 10 \
    --batch_size_training 4 \
    --use_peft true \
    --test_run \
    --save_embedding true \
    --train_data_folder ami_phoneme \