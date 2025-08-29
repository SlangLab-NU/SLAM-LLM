#!/bin/bash -l
#SBATCH -N 1
#SBATCH -c 12
#SBATCH -p gpu
#SBATCH --mem=16GB
#SBATCH --gres=gpu:t4:1
#SBATCH --time=08:00:00
#SBATCH --output=log/%j.output
#SBATCH --error=log/%j.error

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1
export HYDRA_FULL_ERROR=1

nvidia-smi
# module purge
# module load discovery
# module load python
# module load anaconda3/2024.06
# module load ffmpeg  # Comment out if not available
source /projects/van-speech-nlp/zhang.jinda1/src/mllm_experiments/slam-llm/venv/bin/activate
which python

# Change to project root directory
cd /projects/van-speech-nlp/zhang.jinda1/src/mllm_experiments/slam-llm/examples/asr_librispeech

exec > >(tee -a last_slam_run.txt) 2>&1


bash scripts/train_eval.sh \
    --task all \
    --encoder_config w2p-wavlm-dual \
    --num_epochs 100 \
    --batch_size_training 2 \
    --use_peft true \
    --train_data_folder test_run \


# bash scripts/train_eval.sh \
#     --llm_name llama32_1b \
#     --task train \
#     --encoder_config w2p-wavlm-dual \
#     --num_epochs 2 \
#     --batch_size_training 2 \
#     --use_peft true \
#     --freeze_encoder2 false \
#     --train_data_folder test_run \



# bash scripts/train_eval.sh \
#     --llm_name llama32_1b \
#     --task test \
#     --encoder_config w2p-wavlm-dual \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --test_run true \
#     --train_data_folder ami_phoneme \


# bash scripts/train_eval.sh \
#     --llm_name llama32_1b \
#     --task train \
#     --encoder_config w2p-wavlm-dual \
#     --num_epochs 2 \
#     --batch_size_training 2 \
#     --use_peft true \
#     --train_data_folder test_run \