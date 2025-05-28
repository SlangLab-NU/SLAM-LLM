#!/bin/bash -l
#SBATCH -N 1
#SBATCH -c 12
#SBATCH -p reservation
#SBATCH --reservation=zhang.jinda1_test
#SBATCH --mem=16GB
#SBATCH --gres=gpu:v100-sxm2:4
#SBATCH --time=12:00:00
#SBATCH --output=log/%j.output
#SBATCH --error=log/%j.error

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1
export HYDRA_FULL_ERROR=1

nvidia-smi
module purge
module load discovery
module load python/3.8.1 
module load anaconda3/3.7 
module load ffmpeg/20190305 
source activate /work/van-speech-nlp/jindaznb/slamenv/
which python

# ========== Output logging ==========
exec > >(tee -a last_slam_run.txt) 2>&1

run_dir=/work/van-speech-nlp/jindaznb/jslpnb/mllm_experiments/slam-llm/baselines
cd $run_dir
code_dir=examples/asr_librispeech

speech_encoder_path=${RUN_DIR}/models/WavLM-Large.pt
llm_path=/work/van-speech-nlp/jindaznb/jslpnb/mllm_experiments/slam-llm/models/Llama-2-7b-chat-hf
train_data_path=/nfs/maziyang.mzy/data/librispeech/librispeech_train_960h.jsonl
val_data_path=/nfs/maziyang.mzy/data/librispeech/librispeech_dev_other.jsonl

output_dir=/root/tmp/vicuna-7b-v1.5-librispeech-linear-steplrwarmupkeep1e-4-wavlm-large-$(date +"%Y%m%d")

hydra_args="
hydra.run.dir=$output_dir \
++model_config.llm_name=vicuna-7b-v1.5 \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=4096 \
++model_config.encoder_name=wavlm \
++model_config.normalize=true \
++dataset_config.normalize=true \
++model_config.encoder_projector_ds_rate=5 \
++model_config.encoder_path=$speech_encoder_path \
++model_config.encoder_dim=1024 \
++model_config.encoder_projector=linear \
++dataset_config.dataset=speech_dataset \
++dataset_config.train_data_path=$train_data_path \
++dataset_config.val_data_path=$val_data_path \
++dataset_config.input_type=raw \
++train_config.model_name=asr \
++train_config.num_epochs=3 \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=true \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=1000 \
++train_config.total_steps=100000 \
++train_config.lr=1e-4 \
++train_config.validation_interval=1000 \
++train_config.batch_size_training=4 \
++train_config.val_batch_size=4 \
++train_config.num_workers_dataloader=2 \
++train_config.output_dir=$output_dir \
++metric=acc \
"

# -m debugpy --listen 5678 --wait-for-client
if [[ $CUDA_VISIBLE_DEVICES != *","* ]]; then
    python -m debugpy --listen 5678 --wait-for-client $code_dir/finetune_asr.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        $hydra_args
else
    torchrun \
        --nnodes 1 \
        --nproc_per_node 2 \
        --master_port=29503 \
        $code_dir/finetune_asr.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        ++train_config.enable_fsdp=false \
        ++train_config.enable_ddp=true \
        ++train_config.use_fp16=true \
        $hydra_args
fi