#!/bin/bash -l

# export PYTHONPATH=/root/whisper:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1
export HYDRA_FULL_ERROR=1


run_dir=/work/van-speech-nlp/jindaznb/jslpnb/mllm_expriments/slam-llm
code_dir=examples/asr_librispeech

encoder_name=w2v2
encoder_dim=1024
input_type=raw
freeze_encoder=true
speech_encoder_path=vitouphy/wav2vec2-xls-r-300m-timit-phoneme

encoder2_dim=0
freeze_encoder2=true

llm_name=phi35
llm_dim=3072
llm_path=${run_dir}/models/Phi-3.5-mini-instruct
use_peft=true

encoder_projector=linear

data=ami-10h
identifier=${data}_${encoder_name}_${llm_name}_${encoder_projector}_phoneme_freeze
echo "Identifier: $identifier"



output_dir=${run_dir}/out/train/${identifier}