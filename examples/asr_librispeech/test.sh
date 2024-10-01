source activate /work/van-speech-nlp/jindaznb/slamenv/

python -m debugpy --listen 5678 --wait-for-client finetune_asr.py \
    --config-path "conf" \
    --config-name "prompt.yaml
