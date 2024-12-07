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

# 45348488
bash train_eval.sh \
    --llm_name llama32_1b \
    --task test \
    --config_file wavlm-mono \
    --num_epochs 2 \
    --batch_size_training 4 \
    --use_peft true \
    --data_folder ami_phoneme \

# 45348890
# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task test \
#     --config_file wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --data_folder ami \

# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task test \
#     --config_file w2p-wavlm-dual \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --data_folder ami \



# 45317094
# bash train_eval.sh \
#     --task all \
#     --prompt_flag phoneme_seperate \
#     --config_file wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 6 \
#     --data_folder ami_phoneme_seperate \

# bash train_eval.sh \
#     --task all \
#     --prompt_flag phoneme_seperate \
#     --config_file wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 6 \
#     --use_peft false \
#     --data_folder ami_phoneme_seperate \

# 45317247
# bash train_eval.sh \
#     --task all \
#     --prompt_flag phoneme_seperate \
#     --config_file w2p-wavlm-dual \
#     --num_epochs 2 \
#     --batch_size_training 6 \
#     --use_peft false \
#     --data_folder ami_phoneme_seperate

# bash train_eval.sh \
#     --task all \
#     --prompt_flag phoneme_seperate \
#     --config_file w2p-wavlm-dual \
#     --num_epochs 2 \
#     --batch_size_training 6 \
#     --use_peft true \
#     --data_folder ami_phoneme_seperate


# 45261613
# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task test \
#     --config_file wavlm-mono \
#     --num_epochs 10 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --data_folder psst_phoneme \
#     --eval_ckpt last \


# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task test \
#     --config_file whisper-mono \
#     --num_epochs 10 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --data_folder psst_phoneme \
#     --eval_ckpt last

# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task test \
#     --config_file w2p-wavlm-dual \
#     --num_epochs 10 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --data_folder psst_phoneme \
#     --eval_ckpt last



# 45247758
# bash train_eval.sh \
#     --llm_name TinyLlama \
#     --task all \
#     --config_file wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --data_folder librispeech-100 \

# 45247766
# bash train_eval.sh \
#     --llm_name TinyLlama \
#     --task test \
#     --config_file w2p-wavlm-dual \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --data_folder librispeech-100 \









# 45107750
# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task test \
#     --config_file w2p-wavlm-dual \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --data_folder ami \



# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task test \
#     --config_file w2p-wavlm-dual \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --data_folder ami \







# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --config_file w2p-wavlm-dual \
#     --num_epochs 10 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --data_folder psst_phoneme \


# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --config_file w2p-mono \
#     --num_epochs 10 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --data_folder psst_phoneme \

# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --config_file whisper-mono \
#     --num_epochs 10 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --data_folder psst_phoneme \


# 45107282 with phi35 to solve repeating issue
# bash train_eval.sh \
#     --llm_name phi35 \
#     --task all \
#     --config_file wavlm-mono \
#     --num_epochs 5 \
#     --batch_size_training 1 \
#     --use_peft true \
#     --data_folder psst_phoneme \

# bash train_eval.sh \
#     --llm_name phi35 \
#     --task all \
#     --config_file w2p-wavlm-dual \
#     --num_epochs 5 \
#     --batch_size_training 1 \
#     --use_peft true \
#     --data_folder psst_phoneme \


# bash train_eval.sh \
#     --llm_name phi35 \
#     --task all \
#     --config_file w2p-mono \
#     --num_epochs 5 \
#     --batch_size_training 1 \
#     --use_peft true \
#     --data_folder psst_phoneme \

# bash train_eval.sh \
#     --llm_name phi35 \
#     --task all \
#     --config_file whisper-mono \
#     --num_epochs 5 \
#     --batch_size_training 1 \
#     --use_peft true \
#     --data_folder psst_phoneme \






# 45083666
# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --config_file wavlm-mono \
#     --num_epochs 10 \
#     --batch_size_training 1 \
#     --use_peft true \
#     --data_folder psst_phoneme \
#     --freeze_encoder false

# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --config_file w2p-wavlm-dual \
#     --num_epochs 10 \
#     --batch_size_training 1 \
#     --use_peft true \
#     --data_folder psst_phoneme \
#     --freeze_encoder false

# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --config_file w2p-mono \
#     --num_epochs 10 \
#     --batch_size_training 1 \
#     --use_peft true \
#     --data_folder psst_phoneme \
#     --freeze_encoder false

# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --config_file whisper-mono \
#     --num_epochs 10 \
#     --batch_size_training 1 \
#     --use_peft true \
#     --data_folder psst_phoneme \
#     --freeze_encoder false







# 45083694
# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --config_file wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --data_folder ami \


# bash train_eval.sh \
#     --llm_name llama32_1b \
#     --task all \
#     --config_file wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft false \
#     --data_folder ami \









# bash train_eval.sh \
#     --task test \
#     --prompt_flag phoneme_seperate \
#     --config_file w2p-wavlm-dual \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft false \
#     --data_folder ami_phoneme_seperate \








