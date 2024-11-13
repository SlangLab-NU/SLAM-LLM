#!/bin/bash -l
run_dir=/work/van-speech-nlp/jindaznb/jslpnb/mllm_experiments/slam-llm
code_dir=examples/asr_librispeech


encoder_name=wavlm
encoder_dim=1024
input_type=raw
speech_encoder_path=${run_dir}/models/WavLM-Large.pt

encoder_projector=linear

encoder2_dim=${encoder2_dim:-0}

echo "speech encoder name: $encoder_name"
echo "speech encoder path: $speech_encoder_path"
echo "speech encoder2 name: $encoder2_name"
echo "speech encoder2 path: $speech_encoder2_path"
echo "llm_path: $llm_path"
echo "use_peft: $use_peft"
echo "use_fp16: $use_fp16"