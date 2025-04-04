find_last_checkpoint() {
    local output_dir=$1
    # Check if the output_dir exists
    if [ -d "$output_dir" ]; then
        # Find the latest checkpoint folder
        latest_ckpt_folder=$(ls "$output_dir" | grep "asr_epoch" | sort -V | tail -n 1)
        
        # Check if a checkpoint folder was found
        if [ -n "$latest_ckpt_folder" ]; then
            ckpt_path="$output_dir/$latest_ckpt_folder/model.pt"
        else
            echo "No checkpoint found in $output_dir"
            return 1
        fi
    else
        echo "$output_dir does not exist"
        return 1
    fi

    # Construct the full checkpoint folder path
    local ckpt_folder="$output_dir/$latest_ckpt_folder"
    echo "ckpt_folder: $ckpt_folder"

    # Extract the epoch and step using regex or set to default values if extraction fails
    if [[ $latest_ckpt_folder =~ asr_epoch_([0-9]+)_step_([0-9]+) ]]; then
        resume_epoch=${BASH_REMATCH[1]}
        resume_step=${BASH_REMATCH[2]}
    else
        resume_epoch=1
        resume_step=0
    fi

    return 0  # Success return code
}



generate_identifier() {
    local train_data_folder="$1"
    local encoder_name="$2"
    local llm_name="$3"
    local encoder_projector="$4"
    local use_peft="$5"
    local encoder2_name="$6"
    local seed="$7"
    local freeze_encoder="$8"
    local freeze_encoder2="$9"
    local encoder_projector_ds_rate="${10}"

    # Initialize identifier with the basic parts
    local identifier="${train_data_folder}_${encoder_name}_${llm_name}_${encoder_projector}"

    # Add "_freeze" if use_peft is false
    if [[ $use_peft == "false" ]]; then
        identifier+="_freeze_llm"
    fi

    # Add "_peft" if use_peft is true
    if [[ $use_peft == "true" ]]; then
        identifier+="_peft"
    fi

    # Check if encoder_name or encoder2_name is "w2p" and add "phoneme_encoder" to the identifier
    if [[ $encoder_name == "w2p" || $encoder2_name == "w2p" ]]; then
        identifier+="_phoneme_encoder"
    fi

    # Add "_seed_" if seed is set
    if [[ -n $seed ]]; then
        identifier+="_seed_${seed}"
    fi

    # Add "_unfreeze_encoder" if freeze_encoder is false
    if [[ $freeze_encoder == "false" ]]; then
        identifier+="_unfreeze_encoder"
    fi

    # Add "_unfreeze_encoder2" if freeze_encoder2 is false
    if [[ $freeze_encoder2 == "false" ]]; then
        identifier+="_unfreeze_encoder2"
    fi

    # Add "_ds_rate_" if encoder_projector_ds_rate is not 5
    if [[ $encoder_projector_ds_rate -ne 5 ]]; then
        identifier+="_ds_rate_${encoder_projector_ds_rate}"
    fi

    # Return the generated identifier
    echo "$identifier"
}



set_llm_params() {
    local llm_name=$1
    local RUN_DIR=$2
    local use_peft=$3
    local llm_dim
    local llm_path
    local freeze_llm

    # Set llm_dim and llm_path based on llm_name
    case "$llm_name" in
        "TinyLlama")
            llm_dim=2048
            llm_path="${RUN_DIR}/models/TinyLlama-1.1B-Chat-v1.0"
            ;;
        "llama32_1b")
            llm_dim=2048
            llm_path="${RUN_DIR}/models/Llama-3.2-1B-Instruct"
            ;;
        "phi35")
            llm_dim=3072
            llm_path="${RUN_DIR}/models/Phi-3.5-mini-instruct"
            ;;
        "vicuna7b")
            llm_dim=4096
            llm_path="${RUN_DIR}/models/vicuna-7b-v1.5"
            ;;
        *)
            echo "Error: Unsupported llm_name: $llm_name"
            return 1
            ;;
    esac

    # Set freeze_llm based on use_peft
    if [ "$use_peft" = true ]; then
        freeze_llm=false
    else
        freeze_llm=true
    fi

    # Return the values
    echo "$llm_dim $llm_path $freeze_llm"
}
