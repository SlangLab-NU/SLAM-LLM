: '
Set up Argument Parser
'
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -t, --task                Specify the task flag (e.g., 'train', 'eval', 'all')."
    echo "  -d, --prompt_flag         Prompt Flag."
    echo "  -c, --config_file         Path to the configuration file."
    echo "  -e, --num_epochs          Number of epochs to train."
    echo "  -b, --batch_size_training Training batch size."
    echo "  -f, --data_folder         Path to the data folder."
    echo "  -p, --use_peft            Use PEFT flag (true/false)."
    echo "      --debug_flag          Debug flag (true/false)."
    echo "      --test_small          Enable testing with a smaller subset of data."
    echo "  -s, --seed                Set the random seed."
    echo "  -l, --llm_name            Specify the language model to use."
    echo "      --freeze_encoder      Freeze the encoder (true/false). Default is true."
    echo "      --help                Display this help message and exit."
    echo ""
    exit 0
}

# Parse arguments using getopt
OPTIONS=$(getopt -o t:d:c:e:b:f:p:s:l: --long task:,prompt_flag:,config_file:,num_epochs:,batch_size_training:,data_folder:,use_peft:,debug_flag:,test_small,llm_name:,seed:,freeze_encoder:,help -- "$@")
if [ $? -ne 0 ]; then
    echo "Failed to parse arguments."
    exit 1
fi

eval set -- "$OPTIONS"

while true; do
    case "$1" in
        -t|--task)
            task_flag=$2
            shift 2
            ;;
        -d|--prompt_flag)
            prompt_flag=$2
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
        -f|--data_folder)
            data_folder=$2
            shift 2
            ;;
        -p|--use_peft)
            use_peft=$2
            shift 2
            ;;
        -s|--seed)
            seed=$2
            shift 2
            ;;
        -l|--llm_name)
            llm_name=$2
            shift 2
            ;;
        --debug_flag)
            debug_flag=$2
            shift 2
            ;;
        --test_small)
            test_small=true
            shift 1
            ;;
        --freeze_encoder)
            freeze_encoder=$2
            shift 2
            ;;
        --help)
            usage
            ;;
        --)
            shift
            break
            ;;
    esac
done

# Ensure all required arguments are provided
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

# Check if llm is set; if not, set it to 'TinyLlama'
if [ -z "$llm_name" ]; then
    echo "Warning: llm is not set. Setting it to 'TinyLlama' by default."
    llm_name="TinyLlama"
fi

if [ -z "$use_peft" ]; then
    use_peft=true
fi
# Set default value for freeze_encoder if not provided
if [ -z "$freeze_encoder" ]; then
    freeze_encoder=true
fi

echo "Configuration:"
echo "Task: $task_flag"
echo "Prompt Flag: $prompt_flag"
echo "Config File: $config_file"
echo "Epochs: $num_epochs"
echo "Batch Size: $batch_size_training"
echo "Data Folder: $data_folder"
echo "Use PEFT: $use_peft"
echo "LLM Name: $llm_name"
echo "Freeze Encoder: $freeze_encoder"


: '
Set up folders and directories
'
RUN_DIR=/work/van-speech-nlp/jindaznb/jslpnb/mllm_experiments/slam-llm
cd $RUN_DIR
code_dir=examples/asr_librispeech


: '
Select specific dataset file to read in based on prompt_flag value
'
dataset_file="src/slam_llm/datasets/speech_dataset.py:get_speech_dataset"


: '
Set up Data Path
'
train_data_path="${RUN_DIR}/data/${data_folder}/train.jsonl"
if [ -f "${RUN_DIR}/data/${data_folder}/validation.jsonl" ]; then
    val_data_path="${RUN_DIR}/data/${data_folder}/validation.jsonl"
elif [ -f "${RUN_DIR}/data/${data_folder}/val.jsonl" ]; then
    val_data_path="${RUN_DIR}/data/${data_folder}/val.jsonl"
fi
if [[ "$test_small" == true ]]; then
    test_data_path="${RUN_DIR}/data/${data_folder}/test_small.jsonl"
else
    test_data_path="${RUN_DIR}/data/${data_folder}/test.jsonl"
fi

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
Read the llm config for model setup.
'
if [[ "$llm_name" == "TinyLlama" ]]; then
    llm_dim=2048
    llm_path="${run_dir}/models/TinyLlama-1.1B-Chat-v1.0"
fi
if [[ "$llm_name" == "llama32_1b" ]]; then
    llm_dim=2048
    llm_path="${run_dir}/models/Llama-3.2-1B-Instruct"
fi
if [[ "$llm_name" == "phi35" ]]; then
    llm_dim=3072
    llm_path="${run_dir}/models/Phi-3.5-mini-instruct"
fi

if [ "$use_peft" = true ]; then
    freeze_llm=false
else
    freeze_llm=true
fi
freeze_encoder2=${freeze_encoder2:-true}

: '
Initial identifier for folder
'
identifier="${data_folder}_${encoder_name}_${llm_name}_${encoder_projector}"
# Add "_freeze" if use_peft is false
if [[ $use_peft == "false" ]]; then
    identifier+="_freeze_llm"
fi
if [[ $use_peft == "true" ]]; then
    identifier+="_peft"
fi
# Check if encoder_name or encoder2_name is "w2p" and add "phoneme_encoder" to the identifier
if [[ $encoder_name == "w2p" || $encoder2_name == "w2p" ]]; then
    identifier+="_phoneme_encoder"
fi
if [[ -n $seed ]]; then
    identifier+="_seed_${seed}"
fi
if [[ $freeze_encoder == "false" ]]; then
    identifier+="_unfreeze_encoder"
fi

echo "Final identifier: $identifier"
output_dir=${RUN_DIR}/out/${identifier}
split="test"
timestamp=$(date +"%Y%m%d_%H%M%S")  # Format: YearMonthDay_HourMinuteSecond
decode_log="${output_dir}/decode_${split}_beam4_${timestamp}"  # test decode log with timestamp


: '
check output dir exists and find last checkpoint to resume
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

ckpt_folder="$output_dir/$latest_ckpt_folder"
echo "ckpt_folder: $ckpt_folder"

# Extract the epoch and step using sed or set to 0 if extraction fails
if [[ $latest_ckpt_folder =~ asr_epoch_([0-9]+)_step_([0-9]+) ]]; then
    resume_epoch=${BASH_REMATCH[1]}
    resume_step=${BASH_REMATCH[2]}
else
    resume_epoch=1
    resume_step=0
fi
# Output the values for debugging or use
echo "Resume epoch: $resume_epoch"
echo "Resume step: $resume_step"



: '
Start Training
'
# -m debugpy --listen 5678 --wait-for-client
if [[ $task_flag == "train" || $task_flag == "all" ]]; then
    # Define the base command
    if [[ "$debug_flag" == true ]]; then
    command="python -m debugpy --listen 5678 --wait-for-client \
        $code_dir/finetune_asr.py"
    else
        command="python \
            $code_dir/finetune_asr.py"
    fi
    command+=" \
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
        ++train_config.freeze_encoder2=$freeze_encoder2 \
        ++train_config.freeze_llm=$freeze_llm \
        ++train_config.batching_strategy=custom \
        ++train_config.warmup_steps=1000 \
        ++train_config.total_steps=100000 \
        ++train_config.lr=1e-4 \
        ++train_config.validation_interval=3000 \
        ++train_config.batch_size_training=$batch_size_training \
        ++train_config.val_batch_size=$batch_size_training \
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
    if [[ "$encoder_name" == "whisper" ]]; then
        command+=" ++dataset_config.input_type=mel"
        command+=" ++dataset_config.mel_size=128"
    else
        command+=" ++dataset_config.input_type=raw"
    fi
    if [ -n "$seed" ]; then
        command+=" ++train_config.seed=$seed"
    fi
    # Adjust validation interval if debug_flag is true
    # if [[ "$debug_flag" == true ]]; then
    #     command+=" ++train_config.validation_interval=2"
    #     command+=" ++train_config.batch_size_training=1"
    #     command+=" ++train_config.val_batch_size=1"
    # fi
    # Execute the command
    eval $command
fi



: '
check output dir exists and find best checkpoint to infernce
'
# Check if the output_dir exists
if [ -d "$output_dir" ]; then
    # Find all checkpoint folders containing "loss" in the folder name
    ckpt_with_loss=$(ls "$output_dir" | grep "asr_epoch" | grep "loss")
    
    if [ -n "$ckpt_with_loss" ]; then
        # Extract and sort by the numeric loss values
        latest_ckpt_folder=$(echo "$ckpt_with_loss" | awk -F'_loss_' '{print $2, $0}' | sort -n | head -n 1 | awk '{print $2}')
        echo "Selected lowest loss checkpoint: $latest_ckpt_folder"
    else
        # Default to selecting the latest checkpoint by epoch if no "loss" folders are found
        latest_ckpt_folder=$(ls "$output_dir" | grep "asr_epoch" | sort -V | tail -n 1)
        echo "Selected latest checkpoint by epoch: $latest_ckpt_folder"
    fi

    # Check if a checkpoint folder was found
    if [ -n "$latest_ckpt_folder" ]; then
        ckpt_path="$output_dir/$latest_ckpt_folder/model.pt"
        echo "Latest file: $ckpt_path"
    else
        echo "No checkpoint found in $output_dir"
    fi
fi

ckpt_folder="$output_dir/$latest_ckpt_folder"
echo "ckpt_folder $ckpt_folder"


: '
Start Inference
'
# -m debugpy --listen 5678 --wait-for-client
if [[ $task_flag == "test" || $task_flag == "all" ]]; then
    if [[ "$debug_flag" == true ]]; then
    command="python -m debugpy --listen 5678 --wait-for-client \
        $code_dir/finetune_asr.py"
    else
        command="python $code_dir/inference_asr_batch.py"
    fi
    command+=" \
        hydra.run.dir=$ckpt_folder \
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
        ++dataset_config.val_data_path=$test_data_path \
        ++dataset_config.inference_mode=true \
        ++dataset_config.file=$dataset_file \
        ++train_config.model_name=asr \
        ++train_config.freeze_encoder=true \
        ++train_config.freeze_llm=true \
        ++train_config.batching_strategy=custom \
        ++train_config.num_epochs=1 \
        ++train_config.val_batch_size=$batch_size_training \
        ++train_config.num_workers_dataloader=1 \
        ++train_config.output_dir=$output_dir \
        ++decode_log=$decode_log \
        ++ckpt_path=$ckpt_folder/model.pt \
        ++log_config.wandb_exp_name=$identifier \
        ++train_config.use_peft=true"
    if [[ "$encoder_name" == "whisper" ]]; then
        command+=" ++dataset_config.input_type=mel"
        command+=" ++dataset_config.mel_size=128"
    else
        command+=" ++dataset_config.input_type=raw"
    fi
    if [ -n "$seed" ]; then
        command+=" ++train_config.seed=$seed"
    fi
    if [ -n "$prompt" ]; then
        command+=" ++dataset_config.prompt=$prompt"
    fi

    eval $command

    if [ "$prompt_flag" == "phoneme_seperate" ]; then
        python "$code_dir/scripts/wer-seperate.py" --folder "$output_dir"
    else
        python "$code_dir/scripts/wer.py" --folder "$output_dir"
    fi
fi