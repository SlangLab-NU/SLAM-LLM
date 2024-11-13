#!/bin/bash -l

# export PYTHONPATH=/root/whisper:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1
export HYDRA_FULL_ERROR=1


run_dir=/work/van-speech-nlp/jindaznb/jslpnb/mllm_experiments/slam-llm
code_dir=examples/asr_librispeech


encoder_name=whisper
encoder_dim=1280
speech_encoder_path=${run_dir}/models/Whisper/large-v3.pt


encoder_projector=linear

# peft_config_target_modules=[o_proj,qkv_proj]
encoder2_dim=${encoder2_dim:-0}

echo "speech encoder name: $encoder_name"
echo "speech encoder path: $speech_encoder_path"
echo "speech encoder2 name: $encoder2_name"
echo "speech encoder2 path: $speech_encoder2_path"
echo "llm_path: $llm_path"
echo "Identifier: $identifier"
echo "use_peft: $use_peft"
echo "use_fp16: $use_fp16"