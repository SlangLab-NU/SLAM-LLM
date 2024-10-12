#!/bin/bash -l
run_dir=/work/van-speech-nlp/jindaznb/jslpnb/mllm_expriments/slam-llm
code_dir=examples/asr_librispeech


encoder_name=wavlm
encoder_dim=1024
input_type=raw
freeze_encoder=true
speech_encoder_path=${run_dir}/models/WavLM-Large.pt

# llm_name=phi2
# llm_dim=2560
# llm_path=${run_dir}/models/phi-2
llm_name=TinyLlama
llm_dim=2048
llm_path=${run_dir}/models/TinyLlama-1.1B-Chat-v1.0
use_fp16=true
if [ "$use_peft" = true ]; then
    freeze_llm=false
else
    freeze_llm=true
fi

encoder_projector=linear

encoder2_dim=${encoder2_dim:-0}
freeze_encoder2=${freeze_encoder2:-false}

echo "speech encoder name: $encoder_name"
echo "speech encoder path: $speech_encoder_path"
echo "speech encoder2 name: $encoder2_name"
echo "speech encoder2 path: $speech_encoder2_path"
echo "llm_path: $llm_path"
echo "use_peft: $use_peft"
echo "use_fp16: $use_fp16"