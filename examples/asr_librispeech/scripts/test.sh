# Define your variables first
llm_name="TinyLlama"
llm_path="/path/to/llm"
ckpt_path="/path/to/checkpoint"
encoder_name="w2v2"
speech_encoder_path="vitouphy/wav2vec2-xls-r-300m-timit-phoneme"
encoder_dim=1024
encoder_projector="linear"
encoder2_name=""
speech_encoder2_path=""
val_data_path="/path/to/validation/data"
output_dir="/path/to/output"
decode_log="/path/to/decode_log"
identifier="experiment_1"
code_dir="/path/to/code"

# Your command with variables used inside quotes
command="python $code_dir/inference_asr_batch.py \
        --config-path 'conf' \
        --config-name 'prompt.yaml' \
        hydra.run.dir=$ckpt_path \
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
        ++model_config.encoder2_name=$encoder2_name \
        ++model_config.encoder2_path=$speech_encoder2_path \
        ++dataset_config.dataset=speech_dataset \
        ++dataset_config.val_data_path=$val_data_path \
        ++dataset_config.input_type=raw \
        ++dataset_config.inference_mode=true \
        ++train_config.model_name=asr \
        ++train_config.freeze_encoder=true \
        ++train_config.freeze_llm=true \
        ++train_config.batching_strategy=custom \
        ++train_config.num_epochs=1 \
        ++train_config.val_batch_size=1 \
        ++train_config.num_workers_dataloader=1 \
        ++train_config.output_dir=$output_dir \
        ++decode_log=$decode_log \
        ++ckpt_path=$ckpt_path/model.pt \
        ++log_config.wandb_exp_name=$identifier"

# Echo the command (optional, to check the expanded command)
echo $command
