#!/bin/bash -l
#SBATCH -N 1
#SBATCH -c 12
#SBATCH -p gpu
#SBATCH --gres=gpu:v100-sxm2:1
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
data_folder=ami-10h
train_data_path=${run_dir}/data/${data_folder}/ami_train.jsonl
val_data_path=${run_dir}/data/${data_folder}/ami_validation.jsonl
num_epochs=6

config_folder="examples/asr_librispeech/scripts/config"
source ${config_folder}/w2p_mono_freeze.sh

if [ "$use_peft" = true ]; then
    freeze_llm=false
else
    freeze_llm=true
fi

echo "speech encoder name: $encoder_name"
echo "speech encoder path: $speech_encoder_path"
echo "speech encoder2 name: $encoder2_name"
echo "speech encoder2 path: $speech_encoder2_path"
echo "llm_path: $llm_path"
echo "Identifier: $identifier"
echo "Checkpoint path: $ckpt_path"

resume_epoch=${resume_epoch:-1}
encoder2_dim=${encoder2_dim:-0}
freeze_encoder2=${freeze_encoder2:-false}
use_peft=${use_peft: -false}
use_fp16=${use_fp16: -true}


# -m debugpy --listen 5678 --wait-for-client
if [[ $CUDA_VISIBLE_DEVICES != *","* ]]; then
    # Define the base command
python -m debugpy --listen 5678 --wait-for-client $code_dir/finetune_asr.py \
        --config-path 'conf' \
        --config-name 'prompt.yaml' \
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
        ++model_config.encoder_dim=$encoder_dim \
        ++model_config.encoder_projector=$encoder_projector \
        ++dataset_config.dataset=speech_dataset \
        ++dataset_config.train_data_path=$train_data_path \
        ++dataset_config.val_data_path=$val_data_path \
        ++dataset_config.input_type=$input_type \
        ++train_config.model_name=asr \
        ++train_config.num_epochs=$num_epochs \
        ++train_config.freeze_encoder=$freeze_encoder \
        ++train_config.freeze_llm=$freeze_llm \
        ++train_config.batching_strategy=custom \
        ++train_config.warmup_steps=1000 \
        ++train_config.total_steps=100000 \
        ++train_config.lr=1e-4 \
        ++train_config.validation_interval=1000 \
        ++train_config.batch_size_training=1 \
        ++train_config.val_batch_size=1 \
        ++train_config.num_workers_dataloader=1 \
        ++train_config.output_dir=$output_dir \
        ++train_config.resume_epoch=$resume_epoch \
        ++train_config.resume_step=$resume_step \
        ++log_config.use_wandb=false \
        ++log_config.wandb_exp_name=$identifier \
        ++ckpt_path=$ckpt_path \
        ++metric=acc
fi