output_dir=/work/van-speech-nlp/jindaznb/jslpnb/mllm_experiments/slam-llm/out/librispeech-100_phoneme_seperate_wavlm_TinyLlama_dual_peft_seperate

# Check if the output_dir exists
if [ -d "$output_dir" ]; then
    # Find all checkpoint folders containing "loss" in the folder name
    ckpt_with_loss=$(ls -l "$output_dir" | grep "asr_epoch" | grep "loss" | awk '{print $9}')
    
    if [ -n "$ckpt_with_loss" ]; then
        # Sort folders by loss value and select the one with the lowest loss
        latest_ckpt_folder=$(echo "$ckpt_with_loss" | sort -t'_' -k8 -n | head -1)
        echo "Selected lowest loss checkpoint: $latest_ckpt_folder"
    else
        # Default to selecting the latest checkpoint by epoch if no "loss" folders are found
        latest_ckpt_folder=$(ls -l "$output_dir" | grep "asr_epoch" | sort -V | tail -1 | awk '{print $9}')
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
echo "ckpt_folder: $ckpt_folder"