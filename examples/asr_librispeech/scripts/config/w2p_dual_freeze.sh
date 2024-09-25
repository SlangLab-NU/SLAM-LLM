#!/bin/bash

# General settings
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export HYDRA_FULL_ERROR=1


run_dir=/work/van-speech-nlp/jindaznb/jslpnb/mllm_expriments/slam-llm
code_dir=examples/asr_librispeech

# Model settings
encoder_name=wavlm
encoder_dim=1024
input_type=raw
freeze_encoder=true
speech_encoder_path=${run_dir}/models/WavLM-Large.pt

encoder2_name=w2v2
encoder2_dim=1024
freeze_encoder2=true
speech_encoder2_path=vitouphy/wav2vec2-xls-r-300m-timit-phoneme

llm_name=TinyLlama
llm_dim=2048
llm_path=${run_dir}/models/TinyLlama-1.1B-Chat-v1.0
use_peft=true

# Dual encoder settings
dual_encoder=true
encoder_projector=dual

# Data settings
data=ami-10h
identifier=${data}_${encoder_name}_${llm_name}_${encoder_projector}_freeze
train_data_path=${run_dir}/data/ami-10h/ami_train.jsonl
val_data_path=${run_dir}/data/ami-10h/ami_validation.jsonl
output_dir=${run_dir}/out/train/${identifier}
ckpt_path=$output_dir


