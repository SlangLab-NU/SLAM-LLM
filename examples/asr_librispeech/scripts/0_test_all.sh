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
#     --llm_name llama32_1b \
#     --task test \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --data_folder ami_phoneme \

# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task test \
#     --encoder_config w2p-wavlm-dual \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --data_folder ami_phoneme \

# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task test \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --data_folder ami \

# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task test \
#     --encoder_config w2p-wavlm-dual \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --data_folder ami \

# 45452900
# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task test \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --data_folder ami_phoneme_seperate \

# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task test \
#     --encoder_config w2p-wavlm-dual \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --data_folder ami_phoneme_seperate \






# bash train_eval.sh \
#     --task test \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --data_folder librispeech-100 \

# bash train_eval.sh \
#     --task test \
#     --encoder_config w2p-wavlm-dual \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --data_folder librispeech-100 \

# bash train_eval.sh \
#     --task test \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --data_folder librispeech-100_phoneme_only \

# bash train_eval.sh \
#     --task test \
#     --encoder_config w2p-wavlm-dual \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --data_folder librispeech-100_phoneme_only \

# 45452852
bash train_eval.sh \
    --task test \
    --encoder_config wavlm-mono \
    --num_epochs 2 \
    --batch_size_training 4 \
    --use_peft true \
    --debug_flag true \
    --data_folder librispeech-100_phoneme_seperate \


# bash train_eval.sh \
#     --task test \
#     --encoder_config w2p-wavlm-dual \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --data_folder librispeech-100_phoneme_seperate \





# 45406606

# bash train_eval.sh \
#     --task test \
#     --encoder_config w2p-wavlm-dual \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --eval_ckpt best \ 
#     --data_folder psst_phoneme \

# bash train_eval.sh \
#     --task test \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --eval_ckpt best \
#     --data_folder psst_phoneme \

# bash train_eval.sh \
#     --task test \
#     --encoder_config whisper-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --eval_ckpt best \
#     --data_folder psst_phoneme \




# bash train_eval.sh \
#     --task test \
#     --encoder_config w2p-wavlm-dual \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --eval_ckpt last \ 
#     --data_folder psst_phoneme \

# bash train_eval.sh \
#     --task test \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --eval_ckpt last \
#     --data_folder psst_phoneme \

# bash train_eval.sh \
#     --task test \
#     --encoder_config whisper-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --eval_ckpt last \
#     --data_folder psst_phoneme \