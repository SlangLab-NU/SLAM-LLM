#!/bin/bash

# Source the .env file from project root
# Load environment variables from .env
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/../../.."

if [ -f "$PROJECT_ROOT/.env" ]; then
    source "$PROJECT_ROOT/.env"
    echo "Successfully loaded RUN_DIR from .env: $RUN_DIR"
else
    echo "Error: .env file not found in project root: $PROJECT_ROOT/.env"
    exit 1
fi

# Set up Argument Parser
source ./utils.sh

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -t, --task                      Specify the task flag (e.g., 'train', 'eval', 'all')."
    echo "  -c, --encoder_config            Path to the configuration file."
    echo "  -e, --num_epochs                Number of epochs to train."
    echo "  -b, --batch_size_training       Training batch size."
    echo "  -f, --train_data_folder         Path to the training data folder."
    echo "      --test_data_folder          Path to the test data folder (optional, defaults to the same as train_data_folder)."
    echo "  -p, --use_peft                  Use PEFT flag (true/false)."
    echo "      --debug                     Debug flag (true/false)."
    echo "      --test_run                   Enable testing with a smaller subset of data."
    echo "  -s, --seed                      Set the random seed."
    echo "  -l, --llm_name                  Specify the language model to use."
    echo "      --freeze_encoder            Freeze the encoder (true/false). Default is true."
    echo "      --freeze_encoder2           Freeze the second encoder (true/false)."
    echo "      --eval_ckpt                 Set evaluation checkpoint ('last' or 'best'). Default is 'last'."
    echo "      --encoder_projector         Specify the encoder projector to use. Options:"
    echo "                                   'linear'          - Use a linear projector."
    echo "                                   'cov1d-linear'    - Use a 1D convolutional linear projector."
    echo "                                   'q-former'        - Use a QFormer-based projector."
    echo "                                   'dual'     - Use a dual linear concatenation projector."
    echo "                                   Default: 'linear'."
    echo "      --encoder_projector_ds_rate Set the encoder projector down-sampling rate (default is 5)."
    echo "      --save_embedding            Save embeddings during inference (true/false). Default is false."
    echo "      --projector_transfer_learning Use projector transfer learning (true/false). Default is false."
    echo "      --transfer_data_folder       Specify the folder for transfer data (optional)."
    echo "      --llm_inference_config       Specify the inference config file. Default is 'repetition_penalty.json'."
    echo "      --help                      Display this help message and exit."
    echo ""
    exit 0
}

# Parse arguments using getopt
OPTIONS=$(getopt -o t:c:e:b:f:p:s:l: --long task:,encoder_config:,num_epochs:,batch_size_training:,train_data_folder:,test_data_folder:,use_peft:,debug:,test_run,llm_name:,seed:,freeze_encoder:,freeze_encoder2:,eval_ckpt:,encoder_projector:,encoder_projector_ds_rate:,save_embedding:,projector_transfer_learning:,transfer_data_folder:,llm_inference_config:,separate,help -- "$@")
if [ $? -ne 0 ]; then
    echo "Failed to parse arguments."
    exit 1
fi

eval set -- "$OPTIONS"

# Default values
llm_name="llama32_1b"
use_peft=true
freeze_encoder=true
freeze_encoder2=true  # Default value for freeze_encoder2
eval_ckpt="best"
encoder_projector="linear"
encoder_projector_ds_rate=5
save_embedding=false
projector_transfer_learning=false
transfer_data_folder=""
llm_inference_config="repetition_penalty"
separate=false

while true; do
    case "$1" in
        -t|--task)
            task=$2
            shift 2
            ;;
        -c|--encoder_config)
            encoder_config=$2
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
        -f|--train_data_folder)
            train_data_folder=$2
            shift 2
            ;;
        --test_data_folder)
            test_data_folder=$2
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
        --debug)
            debug=true
            shift 2
            ;;
        --test_run)
            test_run=true
            shift 1
            ;;
        --freeze_encoder)
            freeze_encoder=$2
            shift 2
            ;;
        --freeze_encoder2)
            freeze_encoder2=$2
            shift 2
            ;;
        --eval_ckpt)
            eval_ckpt=$2
            shift 2
            ;;
        --encoder_projector)
            encoder_projector=$2
            shift 2
            ;;
        --encoder_projector_ds_rate)
            encoder_projector_ds_rate=$2
            shift 2
            ;;
        --save_embedding)
            save_embedding=$2
            shift 2
            ;;
        --projector_transfer_learning)
            projector_transfer_learning=$2
            shift 2
            ;;
        --transfer_data_folder)
            transfer_data_folder=$2
            shift 2
            ;;
        --llm_inference_config)
            llm_inference_config=$2
            shift 2
            ;;
        --help)
            usage
            ;;
        --separate)
            separate=true
            shift 1
            ;;
        --)
            shift
            break
            ;;
    esac
done


# If test_data_folder is not provided, default it to train_data_folder
if [ -z "$test_data_folder" ]; then
    test_data_folder=$train_data_folder
fi


# Print out configurations.
echo "task: $task"
echo "train_data_folder: $train_data_folder"
echo "test_data_folder: $test_data_folder"
echo "use_peft: $use_peft"
echo "freeze_encoder: $freeze_encoder"
echo "projector_transfer_learning: $projector_transfer_learning"
echo "transfer_data_folder: $transfer_data_folder"
echo "llm_inference_config: $llm_inference_config"
echo "eval_ckpt: $eval_ckpt"


# Set up folders and directories
cd $RUN_DIR
code_dir=examples/asr_librispeech
dataset_file="src/slam_llm/datasets/speech_dataset.py:get_speech_dataset"


# Set up Data Path for train and eval.
train_data_path="${RUN_DIR}/data/${train_data_folder}/train.jsonl"
val_data_path="${RUN_DIR}/data/${train_data_folder}/validation.jsonl"
test_data_path="${RUN_DIR}/data/${test_data_folder}/test.jsonl"

echo "train_data_path: $train_data_path"
echo "val_data_path: $val_data_path"
echo "test_data_path: $test_data_path"



# Read the encoder config for model setup.
# Main script
if [ "$encoder_config" == "w2p-mono" ]; then
    encoder_name="w2v2"
    encoder_dim=1024
    input_type="raw"
    speech_encoder_path="vitouphy/wav2vec2-xls-r-300m-timit-phoneme"
    # Set encoder2_dim to 0 if it is not already set
    encoder2_dim=${encoder2_dim:-0}
fi
if [ "$encoder_config" == "wavlm-mono" ]; then
    encoder_name=wavlm
    encoder_dim=1024
    input_type=raw
    speech_encoder_path=${RUN_DIR}/models/WavLM-Large.pt
    encoder2_dim=${encoder2_dim:-0}
fi
if [ "$encoder_config" == "whisper-mono" ]; then
    encoder_name=whisper
    encoder_dim=1280
    speech_encoder_path=${RUN_DIR}/models/Whisper/large-v3.pt

    # peft_config_target_modules=[o_proj,qkv_proj]
    encoder2_dim=${encoder2_dim:-0}
fi
if [ "$encoder_config" == "w2p-wavlm-dual" ]; then
    encoder_name=wavlm
    encoder_dim=1024
    input_type=raw
    speech_encoder_path=${RUN_DIR}/models/WavLM-Large.pt
    encoder2_name=w2v2
    encoder2_dim=1024
    speech_encoder2_path=vitouphy/wav2vec2-xls-r-300m-timit-phoneme
    encoder_projector=dual
    encoder2_dim=${encoder2_dim:-0}
fi
if [ "$encoder_config" == "w2p-dual-none" ]; then
    encoder_name=wavlm
    encoder_dim=1024
    input_type=raw
    speech_encoder_path=${RUN_DIR}/models/WavLM-Large.pt
    encoder2_name=w2v2
    encoder2_dim=1024
    speech_encoder2_path=vitouphy/wav2vec2-xls-r-300m-timit-phoneme
    encoder_projector=None
    encoder2_dim=${encoder2_dim:-0}
fi
if [ "$encoder_config" == "whisper-dual" ]; then
    encoder_name=whisper
    encoder_dim=1280
    speech_encoder_path=${run_dir}/models/Whisper/large-v3.pt
    encoder2_name=w2v2
    encoder2_dim=1024
    speech_encoder2_path=vitouphy/wav2vec2-xls-r-300m-timit-phoneme
    encoder_projector=dual
    encoder2_dim=${encoder2_dim:-0}
fi


# Read the llm config for model setup.
params=$(set_llm_params "$llm_name" "$RUN_DIR" "$use_peft")
# Parse the returned parameters
llm_dim=$(echo $params | cut -d ' ' -f 1)
llm_path=$(echo $params | cut -d ' ' -f 2)
freeze_llm=$(echo $params | cut -d ' ' -f 3)
# Output the results
echo "LLM Dim: $llm_dim"
echo "LLM Path: $llm_path"
echo "Freeze LLM: $freeze_llm"


# Initial identifier for folder
identifier=$(generate_identifier "$encoder_name" "$llm_name" "$encoder_projector" \
                                "$use_peft" "$encoder2_name" "$seed" "$freeze_encoder" \
                                "$freeze_encoder2" "$encoder_projector_ds_rate")

printf '%*s\n' 50 | tr ' ' '-'
echo "Final identifier: $identifier"
printf '%*s\n' 50 | tr ' ' '-'



# Setup output folder
output_dir=${RUN_DIR}/out/${train_data_folder}/${identifier}
split="test"
timestamp=$(date +"%Y%m%d_%H%M%S")  # Format: YearMonthDay_HourMinuteSecond
decode_log="${output_dir}/decode_${identifier}_beam4"  # test decode log with timestamp

ckpt_path=""
resume_epoch=0
resume_step=0
find_last_checkpoint "$output_dir"

# Output the results
if [ $? -eq 0 ]; then
    echo "Resuming from epoch: $resume_epoch, step: $resume_step"
    echo "Checkpoint Path: $ckpt_path"
else
    echo "Failed to find the latest checkpoint."
fi




# Update configs for transfer learning settings.
# If transfer learning flag is true, set resume_epoch and resume_step to 1 and 0
if [[ "$projector_transfer_learning" == "true" ]]; then
    train_data_path="${RUN_DIR}/data/${transfer_data_folder}/train.jsonl"
    if [ -f "${RUN_DIR}/data/${transfer_data_folder}/validation.jsonl" ]; then
        val_data_path="${RUN_DIR}/data/${transfer_data_folder}/validation.jsonl"
    elif [ -f "${RUN_DIR}/data/${transfer_data_folder}/val.jsonl" ]; then
        val_data_path="${RUN_DIR}/data/${transfer_data_folder}/val.jsonl"
    fi

    # Only reassign test_data_path if it does NOT contain "separate"
    if [[ "$test_data_path" != *separate* ]]; then
        test_data_path="${RUN_DIR}/data/${transfer_data_folder}/test.jsonl"
    fi

    identifier+="_transfer_${transfer_data_folder}"
    output_dir="${RUN_DIR}/out/${train_data_folder}/${identifier}"

    # Try to find the last checkpoint
    if ! find_last_checkpoint "$output_dir"; then
        # If no checkpoint found, set default values
        resume_epoch=1
        resume_step=0
        echo "No checkpoint found in transfer learning folder, starting from epoch 1, step 0"
    else
        # If checkpoint found, print the path
        echo "Found checkpoint at transfer learning folder: $ckpt_path"
    fi


    if find "$output_dir" -type d -name "*epoch*" | grep -q .; then
        echo "Finding checkpoints inside transfer folder"
        find_last_checkpoint "$output_dir"
    fi
    decode_log="${output_dir}/decode_${identifier}_beam4"  # update decode dir for projector transfer learning

    # Print transfer learning information
    echo -e "\n\n\n----- Transfer Learning Information -----"
    echo "Resume Epoch: $resume_epoch"
    echo "Resume Step: $resume_step"
    echo "Train Data Path: $train_data_path"
    echo "Validation Data Path: ${val_data_path:-'Not specified'}"
    echo "Test Data Path: $test_data_path"
    echo "Identifier: $identifier"
    echo "Output Directory: $output_dir"
    echo "----------------------------------------"
    echo "----------------------------------------"
fi



# Start Training
# -m debugpy --listen 5678 --wait-for-client
if [[ $task == "train" || $task == "all" ]]; then
    # Get number of available GPUs
    n_gpus=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
    echo "Detected $n_gpus GPUs available"

    # ========== Start time ==========
    start_time=$(date +%s)
    echo "Start time: $(date)"

    # Define the base command
    if [[ "$debug" == true ]]; then
        command="python -m debugpy --listen 5678 --wait-for-client \
            $code_dir/finetune_asr.py"
    else
        command="torchrun \
            --nnodes 1 \
            --nproc_per_node $n_gpus \
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
        ++model_config.encoder_path=$speech_encoder_path \
        ++model_config.encoder2_name=$encoder2_name \
        ++model_config.encoder2_path=$speech_encoder2_path \
        ++model_config.encoder_dim=$encoder_dim \
        ++model_config.encoder_projector=$encoder_projector \
        ++model_config.encoder_projector_ds_rate=$encoder_projector_ds_rate \
        ++model_config.identifier=$identifier \
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
    if [[ "$encoder_name" == *"whisper"* ]]; then
        command+=" ++dataset_config.input_type=mel"
        command+=" ++dataset_config.mel_size=128"
    else
        command+=" ++dataset_config.input_type=raw"
    fi
    if [ -n "$seed" ]; then
        command+=" ++train_config.seed=$seed"
    fi
    if [[ "$debug" == true || "$train_data_folder" == "test_run" ]]; then
        command+=" ++train_config.validation_interval=2"
        command+=" ++train_config.batch_size_training=1"
        command+=" ++train_config.val_batch_size=1"
    fi
    # Execute the command
    eval $command

    # ========== End time and duration ==========
    end_time=$(date +%s)
    echo "End time: $(date)"

    elapsed=$((end_time - start_time))
    echo "Total run time: ${elapsed}s"
fi



# check output dir exists and find best checkpoint to infernce
# Check if the output_dir exists
if [ -d "$output_dir" ]; then
    if [ "$eval_ckpt" = "best" ]; then
        # Find all checkpoint folders containing "loss" in the folder name
        ckpt_with_loss=$(ls "$output_dir" | grep "asr_epoch" | grep "loss")
        
        if [ -n "$ckpt_with_loss" ]; then
            # Extract and sort by the numeric loss values
            latest_ckpt_folder=$(echo "$ckpt_with_loss" | awk -F'_loss_' '{print $2, $0}' | sort -n | head -n 1 | awk '{print $2}')
            echo "Selected lowest loss checkpoint: $latest_ckpt_folder"
        else
            echo "No checkpoints with loss found. Selecting the checkpoint with the latest epoch and step."

            # Find the checkpoint with the latest epoch and step (e.g., "asr_epoch_2_step_9731")
            latest_ckpt_folder=$(ls "$output_dir" | grep "asr_epoch" | sort -t_ -k3,3n -k5,5n | tail -n 1)
            echo "Selected checkpoint with latest epoch and step: $latest_ckpt_folder"
        fi
    elif [ "$eval_ckpt" = "last" ]; then
        # Default to selecting the latest checkpoint by epoch
        latest_ckpt_folder=$(ls "$output_dir" | grep "asr_epoch" | sort -V | tail -n 1)
        echo "Selected latest checkpoint by epoch: $latest_ckpt_folder"
    else
        echo "Invalid eval_ckpt flag. Use 'best' or 'last'."
        exit 1
    fi

    # Check if a checkpoint folder was found
    if [ -n "$latest_ckpt_folder" ]; then
        ckpt_path="$output_dir/$latest_ckpt_folder/model.pt"
        echo "Checkpoint file: $ckpt_path"
    else
        echo "No checkpoint found in $output_dir"
        exit 1
    fi
else
    echo "Output directory does not exist: $output_dir"
    exit 1
fi

ckpt_folder="$output_dir/$latest_ckpt_folder"
echo "ckpt_folder $ckpt_folder"


# Start Inference
# -m debugpy --listen 5678 --wait-for-client
if [[ $task == "test" || $task == "all" ]]; then
    if [[ "$debug" == true ]]; then
    command="python -m debugpy --listen 5678 --wait-for-client \
        $code_dir/inference_asr_batch.py"
    else
        command="python $code_dir/inference_asr_batch.py"
    fi
    command+=" \
        hydra.run.dir=$output_dir \
        ++model_config.llm_name=$llm_name \
        ++model_config.llm_path=$llm_path \
        ++model_config.llm_dim=$llm_dim \
        ++model_config.encoder_name=$encoder_name \
        ++model_config.normalize=true \
        ++dataset_config.normalize=true \
        ++model_config.encoder_path=$speech_encoder_path \
        ++model_config.encoder_dim=$encoder_dim \
        ++model_config.encoder_projector=$encoder_projector \
        ++model_config.encoder_projector_ds_rate=$encoder_projector_ds_rate \
        ++model_config.encoder2_name=$encoder2_name \
        ++model_config.encoder2_path=$speech_encoder2_path \
        ++model_config.identifier=$identifier \
        ++model_config.llm_inference_config=$llm_inference_config \
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
    # Check if save_embedding is true
    if [[ "$save_embedding" == true ]]; then
        command+=" ++train_config.save_embedding=true"
    fi
    if [[ "$encoder_name" == *"whisper"* ]]; then
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

    if [[ "$test_data_folder" == *"phoneme_separate"* || "$separate" == true ]]; then
        python "$code_dir/scripts/wer_ci.py" --folder "$output_dir" --separate
    else
        python "$code_dir/scripts/wer_ci.py" --folder "$output_dir"
    fi
fi