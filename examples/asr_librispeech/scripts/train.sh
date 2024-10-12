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
export RUN_DIR=/work/van-speech-nlp/jindaznb/jslpnb/mllm_expriments/slam-llm

module purge
module load discovery
module load python/3.8.1 
module load anaconda3/3.7 
module load ffmpeg/20190305 
source activate /work/van-speech-nlp/jindaznb/slamenv/


cd $RUN_DIR
code_dir=examples/asr_librispeech
data_folder=librispeech-100_phoneme


multitask_flag=true  # Change this to false to switch to the non-multitask path
# Conditional logic to set dataset_file based on the flag
if [ "$multitask_flag" = true ]; then
    dataset_file="examples/asr_librispeech/dataset/multitask_speech_dataset.py:get_speech_dataset"
else
    dataset_file="examples/asr_librispeech/model/slam_model_asr.py:model_factory"
fi

# dataset_file="src/slam_llm/datasets/speech_dataset.py:get_speech_dataset"
train_data_path="${RUN_DIR}/data/${data_folder}/${data_folder}_train.jsonl"
val_data_path="${RUN_DIR}/data/${data_folder}/${data_folder}_val.jsonl"
num_epochs=4
batch_size_training=4

config_folder="examples/asr_librispeech/scripts/config"
source ${config_folder}/wavlm-mono.sh

if [ "$multitask_flag" = true ]; then
    output_dir=${RUN_DIR}/out/train/${identifier}_multitask
else
    output_dir=${RUN_DIR}/out/train/${identifier}
fi

# Check if the output_dir exists
if [ -d "$output_dir" ]; then
    # Find the latest checkpoint folder
    latest_ckpt_folder=$(ls -l "$output_dir" | grep "asr_epoch" | sort -V | tail -1 | awk '{print $9}')
    
    # Check if a checkpoint folder was found
    if [ -n "$latest_ckpt_folder" ]; then
        ckpt_path=$output_dir/$latest_ckpt_folder/model.pt
        echo "Latest file: $ckpt_path"
    else
        echo "No checkpoint found in $output_dir"
    fi
else
    echo "$output_dir does not exist"
fi

# Extract the epoch and step using sed or set to 0 if extraction fails
if [[ $latest_ckpt_folder =~ asr_epoch_([0-9]+)_step_([0-9]+) ]]; then
    resume_epoch=${BASH_REMATCH[1]}
    resume_step=${BASH_REMATCH[2]}
else
    resume_epoch=0
    resume_step=0
fi
# Output the values for debugging or use
echo "Resume epoch: $resume_epoch"
echo "Resume step: $resume_step"


# -m debugpy --listen 5678 --wait-for-client
if [[ $CUDA_VISIBLE_DEVICES != *","* ]]; then
    # Define the base command
    command="python $code_dir/finetune_asr.py \
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
        ++dataset_config.file=$dataset_file \
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
        ++train_config.batch_size_training=$batch_size_training \
        ++train_config.val_batch_size=1 \
        ++train_config.num_workers_dataloader=1 \
        ++train_config.output_dir=$output_dir \
        ++train_config.use_fp16=$use_fp16 \
        ++train_config.use_peft=$use_peft \
        ++train_config.resume_epoch=$resume_epoch \
        ++train_config.resume_step=$resume_step \
        ++log_config.use_wandb=true \
        ++log_config.wandb_exp_name=$identifier"

    # Check if ckpt_path is not empty
    if [ -n "$ckpt_path" ]; then
        command+=" ++ckpt_path=$ckpt_path"
    fi

    # Check if peft_config.target_modules is not empty
    if [ -n "$peft_config_target_modules" ]; then
        command+=" ++peft_config.target_modules=$peft_config_target_modules"
    fi

    # Execute the command
    eval $command
fi