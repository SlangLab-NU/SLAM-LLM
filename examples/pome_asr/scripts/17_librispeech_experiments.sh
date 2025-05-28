#!/bin/bash -l
#SBATCH -N 1
#SBATCH -c 12
#SBATCH -p gpu
#SBATCH --mem=16GB
#SBATCH --gres=gpu:v100-sxm2:1
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

exec > >(tee -a last_slam_run.txt) 2>&1


# bash train_eval.sh \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --eval_ckpt best \
#     --train_data_folder librispeech-100_phoneme_separate \


bash train_eval.sh \
    --task all \
    --encoder_config wavlm-mono \
    --num_epochs 2 \
    --batch_size_training 4 \
    --use_peft true \
    --eval_ckpt last \
    --train_data_folder librispeech-100_phoneme_separate \