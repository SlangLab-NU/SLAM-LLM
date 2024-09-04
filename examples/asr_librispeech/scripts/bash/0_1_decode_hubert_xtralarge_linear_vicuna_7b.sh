#!/bin/bash
#export PYTHONPATH=/root/whisper:$PYTHONPATH
# export PYTHONPATH=/root/fairseq:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1 


module purge
module load ffmpeg/20190305 
module load cuda/11.8 
module load discovery/2019-02-21
module load git/2.19.0
module load singularity/3.5.3 

source activate /work/van-speech-nlp/jindaznb/slamenv/

test_speaker="M03"
encoder_name="hubert"

run_dir=/work/van-speech-nlp/jindaznb/jslpnb/mllm_expriments/slam-llm
cd $run_dir
code_dir=examples/asr_librispeech

speech_encoder_path=${run_dir}/models/hubert_xtralarge_ll60k_finetune_ls960.pt
llm_path=${run_dir}/models/vicuna-7b-v1.5

output_dir=${run_dir}/experiments_hubert/vicuna-7b-v1.5-hubert_xtralarge_ll60k_finetune_ls960

ckpt_path=${run_dir}/models

split=test
val_data_path=${run_dir}/data/${test_speaker}_${split}.jsonl

decode_log=${run_dir}/out/experiments_hubert/decode_${test_speaker}_${split}_${encoder_name}_beam4

# -m debugpy --listen 5678 --wait-for-client
python -m debugpy --listen 5678 --wait-for-client \
        $code_dir/inference_asr_batch.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        hydra.run.dir=$ckpt_path \
        ++model_config.llm_name="vicuna-7b-v1.5" \
        ++model_config.llm_path=$llm_path \
        ++model_config.llm_dim=4096 \
        ++model_config.encoder_name=hubert \
        ++model_config.normalize=true \
        ++dataset_config.normalize=true \
        ++model_config.encoder_projector_ds_rate=5 \
        ++model_config.encoder_path=$speech_encoder_path \
        ++model_config.encoder_dim=1280 \
        ++model_config.encoder_type=finetune \
        ++model_config.encoder_projector=linear \
        ++dataset_config.dataset=speech_dataset \
        ++dataset_config.val_data_path=$val_data_path \
        ++dataset_config.input_type=raw \
        ++dataset_config.inference_mode=true \
        ++dataset_config.prompt="Transcribe speech to text. " \
        ++train_config.model_name=asr \
        ++train_config.freeze_encoder=true \
        ++train_config.freeze_llm=true \
        ++train_config.batching_strategy=custom \
        ++train_config.num_epochs=1 \
        ++train_config.val_batch_size=1 \
        ++train_config.num_workers_dataloader=0 \
        ++train_config.output_dir=$output_dir \
        ++decode_log=$decode_log \
        ++ckpt_path=$ckpt_path/hubert_linear_model.pt \
        # ++peft_ckpt=$ckpt_path \
        # ++train_config.use_peft=true \
        # ++train_config.peft_config.r=32 \
        # ++dataset_config.normalize=true \
        # ++model_config.encoder_projector=q-former \
        # ++dataset_config.fix_length_audio=64 \
