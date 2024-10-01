#!/bin/bash -l

# export PYTHONPATH=/root/whisper:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1
export HYDRA_FULL_ERROR=1


run_dir=/work/van-speech-nlp/jindaznb/jslpnb/mllm_expriments/slam-llm
code_dir=examples/asr_librispeech

encoder_name=wavlm
encoder_dim=1024
input_type=raw
freeze_encoder=true
speech_encoder_path=${run_dir}/models/WavLM-Large.pt


llm_name=TinyLlama
llm_dim=2048
llm_path=${run_dir}/models/TinyLlama-1.1B-Chat-v1.0
use_peft=true

encoder_projector=linear

data=ami-10h
identifier=${data_folder}_${encoder_name}_${llm_name}_${encoder_projector}
echo "Identifier: $identifier"

output_dir=${run_dir}/out/train/${identifier}

latest_ckpt_folder=$(ls -l "$output_dir" | grep "asr_epoch" | sort -V | tail -1 | awk '{print $9}')
ckpt_path=$output_dir/$latest_ckpt_folder
echo "Latest file: $ckpt_path"

# Extract the epoch and step using sed or set to 0 if extraction fails
if [[ $latest_ckpt_folder =~ asr_epoch_([0-9]+)_step_([0-9]+) ]]; then
    resume_epoch=${BASH_REMATCH[1]}
    resume_step=${BASH_REMATCH[2]}
else
    resume_epoch=0
    resume_step=0
fi

# Output the values for debugging or use
echo "Resume epoch: $resume_epoch"
echo "Resume step: $resume_step"