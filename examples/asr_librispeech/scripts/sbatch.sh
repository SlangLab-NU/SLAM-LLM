#!/bin/bash -l
#SBATCH -N 1
#SBATCH -c 12
#SBATCH -p gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=08:00:00
#SBATCH --output=log/%j.output
#SBATCH --error=log/%j.error
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jindaznb@gmail.com

# export PYTHONPATH=/root/whisper:$PYTHONPATH
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




# 44709536
# bash run.sh \
#     --task train \
#     --multitask_flag false \
#     --config_file wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --data_folder librispeech-100_phoneme

# 44709525
# bash run.sh \
#     --task train \
#     --multitask_flag true \
#     --config_file wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --data_folder librispeech-100_phoneme