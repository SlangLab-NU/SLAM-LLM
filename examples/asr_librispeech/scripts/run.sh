: '
Set up Argument Parser
'
# # Default values
# task_flag=""
# multitask_flag=false
# config_file=""
# num_epochs=""
# batch_size_training=""
# data_folder=""
# use_peft=false  # Default value for use_peft

# Parse arguments using getopt
OPTIONS=$(getopt -o t:m:c:e:b:d:p: --long task:,multitask_flag:,config_file:,num_epochs:,batch_size_training:,data_folder:,use_peft: -- "$@")
eval set -- "$OPTIONS"

# Extract arguments
while true; do
    case "$1" in
        -t|--task)
            task_flag=$2
            shift 2
            ;;
        -m|--multitask_flag)
            multitask_flag=$2
            shift 2
            ;;
        -c|--config_file)
            config_file=$2
            shift 2
            ;;
        -e|--num_epochs)
            num_epochs=$2
            shift 2
            ;;
        -b|--batch_size_training)
            batch_size_training=$2
            shift 2
            ;;
        -d|--data_folder)
            data_folder=$2
            shift 2
            ;;
        -p|--use_peft)
            use_peft=$2
            shift 2
            ;;
        --)
            shift
            break
            ;;
    esac
done

# Ensure all required arguments are provided
if [ -z "$task_flag" ]; then
    echo "Error: task_flag is required."
    exit 1
fi

if [ -z "$config_file" ]; then
    echo "Error: config_file is required."
    exit 1
fi

if [ -z "$num_epochs" ]; then
    echo "Error: num_epochs is required."
    exit 1
fi

if [ -z "$batch_size_training" ]; then
    echo "Error: batch_size_training is required."
    exit 1
fi

if [ -z "$data_folder" ]; then
    echo "Error: data_folder is required."
    exit 1
fi

# Output the final configuration
echo "Task Flag: $task_flag"
echo "Multitask Flag: $multitask_flag"
echo "Config File: $config_file"
echo "Num Epochs: $num_epochs"
echo "Batch Size Training: $batch_size_training"
echo "Data Folder: $data_folder"
echo "Use PEFT: $use_peft"



: '
Set up folders and directories
'
RUN_DIR=/work/van-speech-nlp/jindaznb/jslpnb/mllm_expriments/slam-llm
cd $RUN_DIR
code_dir=examples/asr_librispeech


: '
Select specific dataset file to read in
'
if [ "$multitask_flag" = true ]; then
    dataset_file="examples/asr_librispeech/dataset/multitask_speech_dataset.py:get_speech_dataset"
else
    dataset_file="src/slam_llm/datasets/speech_dataset.py:get_speech_dataset"
fi



: '
Set up Data Path
'
# dataset_file="src/slam_llm/datasets/speech_dataset.py:get_speech_dataset"
train_data_path="${RUN_DIR}/data/${data_folder}/${data_folder}_train.jsonl"
val_data_path="${RUN_DIR}/data/${data_folder}/${data_folder}_val.jsonl"
test_data_path="${RUN_DIR}/data/${data_folder}/${data_folder}_test.jsonl"
if [[ ! -f "$train_data_path" ]]; then
    echo "Error: Train data path not found at $train_data_path"
    exit 1
elif [[ ! -f "$val_data_path" ]]; then
    echo "Error: Validation data path not found at $val_data_path"
    exit 1
elif [[ ! -f "$test_data_path" ]]; then
    echo "Error: Test data path not found at $test_data_path"
    exit 1
fi


: '
Read the model configuration file for model setup.
'
config_folder="examples/asr_librispeech/scripts/model_config"
source ${config_folder}/${config_file}.sh


: '
Initial identifier for folder
'
identifier="${data_folder}_${encoder_name}_${llm_name}_${encoder_projector}"
# Add "_freeze" if use_peft is false
if [[ $use_peft == "false" ]]; then
    identifier+="_freeze"
fi
if [[ $use_peft == "true" ]]; then
    identifier+="_unfreeze"
fi
# Check if encoder_name or encoder2_name is "w2p" and add "phoneme_encoder" to the identifier
if [[ $encoder_name == "w2p" || $encoder2_name == "w2p" ]]; then
    identifier+="_phoneme_encoder"
fi
if [ "$multitask_flag" = true ]; then
    identifier+='_multitask'
fi
# The identifier will now be modified dynamically based on the conditions
echo "Final identifier: $identifier"
output_dir=${RUN_DIR}/out/train/${identifier}_multitask
decode_log=$output_dir/decode_${split}_beam4 # test decode log



: '
check output dir exists and find latest(best) checkpoint to resume
'
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



: '
Start Training
'
# -m debugpy --listen 5678 --wait-for-client
if [[ $task_flag == "train" ]]; then
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
        ++train_config.use_fp16=true \
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

elif [[ $flag == "test" ]]; then
    command="python $code_dir/inference_asr_batch.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
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
        ++log_config.wandb_exp_name=$identifier \
        ++train_config.use_peft=true"
    eval $command

    python $code_dir/scripts/wer.py \
            --folder $output_dir
fi