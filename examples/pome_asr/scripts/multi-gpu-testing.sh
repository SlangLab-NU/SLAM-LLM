#!/bin/bash -l
#SBATCH -N 1
#SBATCH -c 12
#SBATCH -p reservation
#SBATCH --reservation=zhang.jinda1_test
#SBATCH --mem=16GB
#SBATCH --gres=gpu:v100-sxm2:4
#SBATCH --time=12:00:00
#SBATCH --output=log/%j.output
#SBATCH --error=log/%j.error

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1
export HYDRA_FULL_ERROR=1

nvidia-smi
module purge
module load discovery
module load python/3.8.1 
module load anaconda3/3.7 
module load ffmpeg/20190305 
source activate /work/van-speech-nlp/jindaznb/slamenv/
which python

exec > >(tee -a last_slam_run.txt) 2>&1

seed=$(head /dev/urandom | tr -dc '0-9' | head -c 10)

# ========== Main training command ==========
bash train_eval.sh \
    --task train \
    --encoder_config wavlm-mono \
    --num_epochs 4 \
    --batch_size_training 1 \
    --use_peft true \
    --eval_ckpt best \
    --seed $seed \
    --train_data_folder test_run \

# bash train_eval.sh \
#     --task all \
#     --encoder_config w2p-wavlm-dual \
#     --num_epochs 100 \
#     --batch_size_training 2 \
#     --use_peft true \
#     --train_data_folder test_run \

