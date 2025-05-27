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
export HUGGINGFACE

module purge
module load discovery
module load python/3.8.1
module load anaconda3/3.7 
module load ffmpeg/20190305 
source activate /work/van-speech-nlp/jindaznb/slamenv/
which python


export RUN_DIR=/work/van-speech-nlp/jindaznb/jslpnb/mllm_experiments/slam-llm
export HF_HOME=/scratch/zhang.jinda1/temp/huggingface_cache
export TRANSFORMERS_CACHE=/scratch/zhang.jinda1/temp/huggingface_cache/models
export HF_DATASETS_CACHE=/scratch/zhang.jinda1/temp/huggingface_cache/datasets



# python wav2vec2.py --dataset ami --pretrained_model microsoft/wavlm-large
# python wav2vec2.py --dataset aphasia --pretrained_model microsoft/wavlm-large
# python wav2vec2.py --dataset librispeech-100 --pretrained_model microsoft/wavlm-large



# python wav2vec2.py --dataset ami
# python wav2vec2.py --dataset aphasia
# python wav2vec2.py --dataset librispeech-100
# python wav2vec2.py --dataset librispeech-100_phoneme



# python wav2vec2.py --dataset test_run

# python multiGPU_testing.py --num_gpus 4 --compare --epochs 10