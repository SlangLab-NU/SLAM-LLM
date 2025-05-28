#!/bin/bash -l

# Load environment variables from .env
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/../../../.."

if [ -f "$PROJECT_ROOT/.env" ]; then
    source "$PROJECT_ROOT/.env"
else
    echo "Error: .env file not found in project root: $PROJECT_ROOT/.env"
    exit 1
fi

run_dir=$RUN_DIR
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