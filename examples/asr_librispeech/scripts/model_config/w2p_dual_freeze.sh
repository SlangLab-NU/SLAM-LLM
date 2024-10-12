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

encoder2_name=w2v2
encoder2_dim=1024
freeze_encoder2=true
speech_encoder2_path=vitouphy/wav2vec2-xls-r-300m-timit-phoneme

# llm_name=phi2
# llm_dim=2560
# llm_path=${run_dir}/models/phi-2
llm_name=TinyLlama
llm_dim=2048
llm_path=${run_dir}/models/TinyLlama-1.1B-Chat-v1.0
use_peft=true
use_fp16=$use_peft
if [ "$use_peft" = true ]; then
    freeze_llm=false
else
    freeze_llm=true
fi

encoder_projector=dual


identifier=${data_folder}_${encoder_name}_${llm_name}_${encoder_projector}_phoneme_freeze

encoder2_dim=${encoder2_dim:-0}
freeze_encoder2=${freeze_encoder2:-false}

echo "speech encoder name: $encoder_name"
echo "speech encoder path: $speech_encoder_path"
echo "speech encoder2 name: $encoder2_name"
echo "speech encoder2 path: $speech_encoder2_path"
echo "llm_path: $llm_path"
echo "Identifier: $identifier"
echo "use_peft: $use_peft"
echo "use_fp16: $use_fp16"