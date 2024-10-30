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



# 44924913
# bash train_eval.sh \
#     --task all \
#     --prompt_flag phoneme_seperate\
#     --config_file w2p-wavlm-dual \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --data_folder ami_phoneme_seperate \


# 44915397 44924888
# bash train_eval.sh \
#     --task all \
#     --config_file w2p-wavlm-dual \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --data_folder ami \



# 44913245
# bash train_eval.sh \
#     --task train \
#     --prompt_flag phoneme_seperate \
#     --config_file wavlm-mono \
#     --num_epochs 6 \
#     --batch_size_training 6 \
#     --use_peft true \
#     --data_folder ami_phoneme_seperate

# 44900294
# bash train_eval.sh \
#     --task all \
#     --config_file w2p-wavlm-dual \
#     --num_epochs 2 \
#     --batch_size_training 6 \
#     --use_peft true \
#     --data_folder ami_phoneme_only



# 44892371
# bash train_eval.sh \
#     --task all \
#     --prompt_flag phoneme_seperate \
#     --config_file wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 6 \
#     --use_peft true \
#     --data_folder ami_phoneme_seperate

# 44892379
# bash train_eval.sh \
#     --task all \
#     --config_file wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 6 \
#     --use_peft true \
#     --data_folder ami_phoneme_only

# 44892383
# bash train_eval.sh \
#     --task all \
#     --config_file wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 6 \
#     --use_peft true \
#     --data_folder ami


# 44892467
# bash train_eval.sh \
#     --task all \
#     --config_file w2p-mono \
#     --num_epochs 2 \
#     --batch_size_training 6 \
#     --use_peft true \
#     --data_folder librispeech-100_phoneme_only``





# ami base 44888595
# bash train_eval.sh \
#     --task all \
#     --prompt_flag separate \
#     --config_file whisper-mono \
#     --num_epochs 2 \
#     --batch_size_training 6 \
#     --use_peft true \
#     --data_folder ami \


# 44889915
# bash train_eval.sh \
#     --task test \
#     --prompt_flag phoneme_seperate\
#     --config_file w2p-wavlm-dual \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --data_folder librispeech-100_phoneme_seperate \



# 44865818 batch size has to be 2
# bash train_eval.sh \
#     --task all \
#     --prompt_flag separate \
#     --config_file whisper-mono \
#     --num_epochs 2 \
#     --batch_size_training 2 \
#     --use_peft true \
#     --data_folder ami_nbest \




# 44832596


# 44832614
# bash train_eval.sh \
#     --task all \
#     --config_file w2p-wavlm-dual \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --data_folder librispeech-100_phoneme_only




# python wer.py --folder /work/van-speech-nlp/jindaznb/jslpnb/mllm_experiments/slam-llm/out/librispeech-100_phoneme_only_wavlm_TinyLlama_linear_peft

# bash train_eval.sh \
#     --task all \
#     --config_file wavlm-mono \
#     --num_epochs 2 \
#     --batch_size_training 4 \
#     --use_peft true \
#     --data_folder librispeech-100_phoneme_only \
#     --llm llama32_1b




