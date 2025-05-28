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


encoder_name=wavlm
encoder_dim=1024
input_type=raw
speech_encoder_path=${run_dir}/models/WavLM-Large.pt

encoder2_name=w2v2
encoder2_dim=1024
speech_encoder2_path=vitouphy/wav2vec2-xls-r-300m-timit-phoneme


encoder_projector=dual


encoder2_dim=${encoder2_dim:-0}

echo "speech encoder name: $encoder_name"
echo "speech encoder path: $speech_encoder_path"
echo "speech encoder2 name: $encoder2_name"
echo "speech encoder2 path: $speech_encoder2_path"
echo "llm_path: $llm_path"
echo "Identifier: $identifier"
echo "use_peft: $use_peft"
echo "use_fp16: $use_fp16"