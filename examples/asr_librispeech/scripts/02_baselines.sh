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
#     --seed 42 \
#     --train_data_folder librispeech-100 \


# bash train_eval.sh \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --eval_ckpt best \
#     --seed 42 \
#     --train_data_folder librispeech-100_phoneme \


# bash train_eval.sh \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --eval_ckpt best \
#     --seed 42 \
#     --train_data_folder librispeech-100_phoneme_separate \



# bash train_eval.sh \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --seed 42 \
#     --train_data_folder ami \



# bash train_eval.sh \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --seed 42 \
#     --train_data_folder ami_phoneme \


# bash train_eval.sh \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --seed 42 \
#     --train_data_folder ami_phoneme_separate \


# bash train_eval.sh \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --seed 42 \
#     --train_data_folder ami_ec \




# bash train_eval.sh \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --seed 42 \
#     --train_data_folder aphasia \


# bash train_eval.sh \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --seed 42 \
#     --train_data_folder aphasia_phoneme \



# bash train_eval.sh \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --seed 42 \
#     --train_data_folder aphasia_phoneme_separate \



#
#
#



# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --train_data_folder librispeech-100 \
#     --projector_transfer_learning true \
#     --transfer_data_folder ami \


# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --train_data_folder librispeech-100 \
#     --projector_transfer_learning true \
#     --transfer_data_folder aphasia \

# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --train_data_folder ami \
#     --projector_transfer_learning true \
#     --transfer_data_folder aphasia \


# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --train_data_folder librispeech-100_phoneme \
#     --projector_transfer_learning true \
#     --transfer_data_folder ami_phoneme \

# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --train_data_folder librispeech-100_phoneme \
#     --projector_transfer_learning true \
#     --transfer_data_folder psst_phoneme \


# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --train_data_folder librispeech-100_phoneme \
#     --projector_transfer_learning true \
#     --transfer_data_folder aphasia_phoneme \



# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --train_data_folder ami_phoneme \
#     --projector_transfer_learning true \
#     --transfer_data_folder psst_phoneme \


# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --train_data_folder ami_phoneme \
#     --projector_transfer_learning true \
#     --transfer_data_folder aphasia_phoneme \

# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --train_data_folder psst_phoneme \
#     --projector_transfer_learning true \
#     --transfer_data_folder aphasia_phoneme \



# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --train_data_folder ami_phoneme \
#     --all_data_folder librispeech-100_phoneme \

# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --train_data_folder psst_phoneme \
#     --all_data_folder librispeech-100_phoneme \

# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --train_data_folder aphasia_phoneme \
#     --all_data_folder librispeech-100_phoneme \


# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --train_data_folder librispeech-100_phoneme \
#     --all_data_folder ami_phoneme \


# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --train_data_folder psst_phoneme \
#     --all_data_folder ami_phoneme \

# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --train_data_folder aphasia_phoneme \
#     --all_data_folder ami_phoneme \


# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --train_data_folder aphasia_phoneme \
#     --all_data_folder psst_phoneme \

# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --train_data_folder librispeech-100_phoneme \
#     --all_data_folder psst_phoneme \


# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --train_data_folder ami_phoneme \
#     --all_data_folder psst_phoneme \


# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --train_data_folder librispeech-100_phoneme \
#     --all_data_folder aphasia_phoneme \


# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --train_data_folder ami_phoneme \
#     --all_data_folder aphasia_phoneme \


# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --train_data_folder psst_phoneme \
#     --all_data_folder aphasia_phoneme \


# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --train_data_folder librispeech-100 \
#     --all_data_folder ami \

# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --train_data_folder librispeech-100 \
#     --all_data_folder aphasia \


# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --train_data_folder ami \
#     --all_data_folder librispeech-100 \


# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --train_data_folder ami \
#     --all_data_folder aphasia \


# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --train_data_folder aphasia \
#     --all_data_folder librispeech-100 \


# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --train_data_folder aphasia \
#     --all_data_folder ami \



# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --train_data_folder psst_phoneme \
#     --encoder_projector_ds_rate 6 


# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --train_data_folder psst_phoneme \
#     --encoder_projector_ds_rate 4


# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --train_data_folder ami \
#     --encoder_projector_ds_rate 6 


# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --train_data_folder ami \
#     --encoder_projector_ds_rate 4



# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --train_data_folder ami_phoneme \
#     --encoder_projector_ds_rate 6 


# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --train_data_folder ami_phoneme \
#     --encoder_projector_ds_rate 4


# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --train_data_folder aphasia_phoneme \
#     --encoder_projector_ds_rate 6 


# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --train_data_folder aphasia_phoneme \
#     --encoder_projector_ds_rate 4

# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --train_data_folder aphasia \
#     --encoder_projector_ds_rate 6 


# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --train_data_folder aphasia \
#     --encoder_projector_ds_rate 4


# # different projector

# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --train_data_folder psst_phoneme \
#     --encoder_projector cov1d-linear \


# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --train_data_folder psst_phoneme \
#     --encoder_projector q-former \


# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --train_data_folder ami \
#     --encoder_projector cov1d-linear \


# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --train_data_folder ami \
#     --encoder_projector q-former \



# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --train_data_folder ami_phoneme \
#     --encoder_projector cov1d-linear \


# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --train_data_folder ami_phoneme \
#     --encoder_projector q-former \


# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --train_data_folder aphasia_phoneme \
#     --encoder_projector cov1d-linear \


# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --train_data_folder aphasia_phoneme \
#     --encoder_projector q-former \

# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --train_data_folder aphasia \
#     --encoder_projector cov1d-linear \


# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --encoder_config wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --train_data_folder aphasia \
#     --encoder_projector q-former \