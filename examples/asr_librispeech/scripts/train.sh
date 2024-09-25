#!/bin/bash -l
#SBATCH -N 1
#SBATCH -c 12
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=log/%j.output
#SBATCH --error=log/%j.error
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jindaznb@gmail.com

# export PYTHONPATH=/root/whisper:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1
export HYDRA_FULL_ERROR=1

module purge
module load python/3.8.1
module load anaconda3/3.7
module load ffmpeg/20190305
source activate /work/van-speech-nlp/jindaznb/slamenv/

run_dir=/work/van-speech-nlp/jindaznb/jslpnb/mllm_expriments/slam-llm
cd $run_dir

code_dir=examples/asr_librispeech
train_data_path=${run_dir}/data/ami-1h/ami_train.jsonl
val_data_path=${run_dir}/data/ami-1h/ami_validation.jsonl

config_folder="examples/asr_librispeech/scripts/config"
source ${config_folder}/w2p_mono_freeze_phi35.sh

echo "speech encoder name: $encoder_name"
echo "speech encoder path: $speech_encoder_path"
echo "speech encoder2 name: $encoder2_name"
echo "speech encoder2 path: $speech_encoder2_path"
echo "llm_path: $llm_path"
echo "Identifier: $identifier"

# -m debugpy --listen 5678 --wait-for-client
if [[ $CUDA_VISIBLE_DEVICES != *","* ]]; then
    python $code_dir/finetune_asr.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        hydra.run.dir=$output_dir \
        ++model_config.llm_name=$llm_name \
        ++model_config.llm_path=$llm_path \
        ++model_config.llm_dim=$llm_dim \
        ++model_config.encoder_name=$encoder_name \
        ++model_config.normalize=true \
        ++dataset_config.normalize=true \
        ++model_config.encoder_projector_ds_rate=5 \
        ++model_config.encoder_path=$speech_encoder_path \
        ++model_config.encoder2_name=$encoder2_name \
        ++model_config.encoder2_path=$speech_encoder2_path \
        ++model_config.encoder2_dim=$encoder2_dim \
        ++model_config.encoder_dim=$encoder_dim \
        ++model_config.encoder_projector=$encoder_projector \
        ++dataset_config.dataset=speech_dataset \
        ++dataset_config.train_data_path=$train_data_path \
        ++dataset_config.val_data_path=$val_data_path \
        ++dataset_config.input_type=$input_type \
        ++train_config.model_name=asr \
        ++train_config.num_epochs=3 \
        ++train_config.freeze_encoder=$freeze_encoder \
        ++train_config.freeze_encoder2=$freeze_encoder2 \
        ++train_config.freeze_llm=true \
        ++train_config.batching_strategy=custom \
        ++train_config.warmup_steps=1000 \
        ++train_config.total_steps=100000 \
        ++train_config.lr=1e-4 \
        ++train_config.validation_interval=1000 \
        ++train_config.batch_size_training=1 \
        ++train_config.val_batch_size=1 \
        ++train_config.num_workers_dataloader=1 \
        ++train_config.output_dir=$output_dir \
        ++train_config.use_fp16=true \
        ++train_config.use_peft=$use_peft \
        ++log_config.use_wandb=false \
        ++log_config.wandb_exp_name=$identifier \
        ++ckpt_path=$ckpt_path \
        ++metric=acc
fi