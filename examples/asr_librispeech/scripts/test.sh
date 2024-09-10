#!/bin/bash
#SBATCH -N 1
#SBATCH -c 3
#SBATCH -p gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=08:00:00
#SBATCH --output=/work/van-speech-nlp/jindaznb/jslpnb/log/%j.output
#SBATCH --error=/work/van-speech-nlp/jindaznb/jslpnb/log/%j.error

# export PYTHONPATH=/root/whisper:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1
export HYDRA_FULL_ERROR=1

module load anaconda3/3.7
module load ffmpeg/20190305
source activate /work/van-speech-nlp/jindaznb/slamenv/

run_dir=/work/van-speech-nlp/jindaznb/jslpnb/mllm_expriments/slam-llm
cd $run_dir
code_dir=examples/asr_librispeech

encoder_name=w2v2
encoder_dim=1024
input_type=raw
speech_encoder_path=facebook/wav2vec2-large-xlsr-53
echo "speech encoder name: $encoder_name"
echo "speech encoder path: $speech_encoder_path"
llm_name=phi-2
llm_dim=2560
llm_path=${run_dir}/models/${llm_name}
echo "llm_path: $llm_path"
dual_encoder=false
encoder_projector=linear

data=ami-10h
identifier=${data}_${encoder_name}_${llm_name}_${encoder_projector}
echo "Identifier: $identifier"

# auto find checkpoint
output_dir=/work/van-speech-nlp/jindaznb/jslpnb/mllm_expriments/slam-llm/out/train/${identifier}
latest_ckpt_folder=$(ls -l "$output_dir" | grep "asr_epoch" | sort -V | tail -1 | awk '{print $9}')
ckpt_path=$output_dir/$latest_ckpt_folder
echo "Latest file: $ckpt_path"

train_data_path=${run_dir}/data/ami-10h/ami_train.jsonl
val_data_path=${run_dir}/data/ami-10h/ami_validation.jsonl

output_dir=${run_dir}/out/train/${identifier}



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
        ++model_config.encoder_dim=$encoder_dim \
        ++model_config.encoder_projector=$encoder_projector \
        ++model_config.dual_encoder=$dual_encoder \
        ++dataset_config.dataset=speech_dataset \
        ++dataset_config.train_data_path=$train_data_path \
        ++dataset_config.val_data_path=$val_data_path \
        ++dataset_config.input_type=$input_type \
        ++train_config.model_name=asr \
        ++train_config.num_epochs=10 \
        ++train_config.freeze_encoder=true \
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
        ++ckpt_path=$ckpt_path/model.pt \
        ++metric=acc
fi