#!/bin/bash



set_encoder_config() {
    RUN_DIR=/work/van-speech-nlp/jindaznb/jslpnb/mllm_experiments/slam-llm
    if [ "$encoder_config" == "w2p-mono" ]; then
        encoder_name="w2v2"
        encoder_dim=1024
        input_type="raw"
        speech_encoder_path="vitouphy/wav2vec2-xls-r-300m-timit-phoneme"
        # Set encoder2_dim to 0 if it is not already set
        encoder2_dim=${encoder2_dim:-0}
    fi
    if [ "$encoder_config" == "wavlm-mono" ]; then
        encoder_name=wavlm
        encoder_dim=1024
        input_type=raw
        speech_encoder_path=${RUN_DIR}/models/WavLM-Large.pt
        encoder2_dim=${encoder2_dim:-0}
    fi
    if [ "$encoder_config" == "whisper-mono" ]; then
        encoder_name=whisper
        encoder_dim=1280
        speech_encoder_path=${RUN_DIR}/models/Whisper/large-v3.pt

        # peft_config_target_modules=[o_proj,qkv_proj]
        encoder2_dim=${encoder2_dim:-0}
    fi
    if [ "$encoder_config" == "w2p-wavlm-dual" ]; then
        encoder_name=wavlm
        encoder_dim=1024
        input_type=raw
        speech_encoder_path=${RUN_DIR}/models/WavLM-Large.pt
        encoder2_name=w2v2
        encoder2_dim=1024
        speech_encoder2_path=vitouphy/wav2vec2-xls-r-300m-timit-phoneme
        encoder_projector=dual_linear
        encoder2_dim=${encoder2_dim:-0}
    fi
}