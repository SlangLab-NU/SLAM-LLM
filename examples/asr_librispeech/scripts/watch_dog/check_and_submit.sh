#!/bin/bash

code_dir=/work/van-speech-nlp/jindaznb/jslpnb/mllm_experiments/slam-llm
checkpoint_dir=$code_dir/data/librispeech
g=20  # Maximum number of epochs after which the script should stop


# Function to find the latest checkpoint and update the job name
update_job_name_and_checkpoint() {
    file_name="librispeech-100h_train-clean-100_hypo.json"
    ckpt_path="$checkpoint_dir/$file_name"

    # Replace '_hypo.json' with '_text.txt' to get the corresponding TXT file name
    txt_file="${ckpt_path/_hypo.json/_text.txt}"

    # Use wc -l to count the number of lines in both JSON and TXT files
    json_line_count=$(wc -l < "$ckpt_path")
    txt_line_count=$(wc -l < "$txt_file")

    echo "The JSON file has $json_line_count lines."
    echo "The TXT file has $txt_line_count lines."
 
    # Check if the JSON file has more lines than the TXT file
    if [ "$json_line_count" -gt "$txt_line_count" ]; then
        echo "Error: The JSON file has more lines than the TXT file."
        exit 0
    fi
    
    job_name="h2t_train_${json_line_count}"
    sed -i "s/#SBATCH --job-name=.*/#SBATCH --job-name=$job_name/" $code_dir/src/h2t_train.bash
}


# Check if there are running or pending jobs with "h2t_submit.bash"
if squeue -u `whoami` | grep -E " R| PD" | grep "h2t" > /dev/null; then
    echo "Job 'h2t_submit.bash' still running or pending in the queue. No action taken."
else
    # Update job name and checkpoints
    update_job_name_and_checkpoint

    # Navigate to the directory containing the SLURM submission script
    cd $code_dir/src
    
    # Submit the next job
    echo "Submitting job $job_name at $(date)"
    sbatch h2t_train.bash
fi







test(){
    # Change to the directory where checkpoints are stored
    cd $checkpoint_dir

    # Find the latest epoch checkpoint file (e.g., epoch-1.pt, epoch-2.pt, ...) and store the latest one.
    # Use 'sort -V' to sort version numbers correctly and 'tail -n 1' to get the last (latest) file.
    local latest_epoch=$(ls epoch-*.pt 2>/dev/null | sort -V | tail -n 1)

    # Find the latest batch checkpoint file (e.g., checkpoint-1.pt, checkpoint-2.pt, ...) in a similar manner.
    local latest_batch=$(ls checkpoint-*.pt 2>/dev/null | sort -V | tail -n 1)

    # Initialize default values:
    # Set epoch_num to 1 to start training from the first epoch if no checkpoints are found.
    local epoch_num=1

    # Set batch_num to 0, meaning training will start from the first batch by default.
    local batch_num=0

    # Initialize job_name as an empty string. This will be used to set the SLURM job name later.
    local job_name=""

    # Determine the current epoch number based on the latest epoch checkpoint:
    if [[ -n $latest_epoch ]]; then  # Check if a valid epoch checkpoint exists.
        # Extract the numeric value from the file name (e.g., "epoch-3.pt" will extract "3").
        epoch_num=$(echo $latest_epoch | grep -o -E '[0-9]+')

        # Increment the extracted epoch number by 1, so training can resume from the next epoch.
        epoch_num=$((epoch_num + 1))
    fi

    # Determine the batch number if the latest checkpoint is a batch checkpoint:
    if [[ -n $latest_batch && ( -z $latest_epoch || $latest_epoch -ot $latest_batch ) ]]; then
        # Check if the latest batch checkpoint exists and is newer than the latest epoch checkpoint.
        # Extract the numeric value from the batch file name (e.g., "checkpoint-7.pt" extracts "7").
        batch_num=$(echo $latest_batch | grep -o -E '[0-9]+')
    fi

    # Set the SLURM job name based on the determined epoch and batch numbers:
    # Example: If epoch_num is 3 and batch_num is 5, the job name will be "train_3_5".
    job_name="train_${epoch_num}_${batch_num}"

    # Check if the maximum number of epochs has been reached:
    if (( epoch_num > max_epochs )); then
        # If the current epoch exceeds max_epochs, print a message and exit.
        echo "Maximum number of epochs ($max_epochs) reached at $(date). No further action required."
        exit 0  # Terminate the script since no further training is needed.
    fi

    # Update the SLURM job script to include the new job name, start epoch, and start batch:
    # Replace the existing SLURM job name in the job script with the new job_name variable.
    sed -i "s/#SBATCH --job-name=.*/#SBATCH --job-name=$job_name/" $code_dir/src/train_job.sh

    # Update the '--start-epoch' parameter in the job script with the determined epoch number.
    sed -i "s/--start-epoch [0-9]*/--start-epoch $epoch_num/" $code_dir/src/train_job.sh

    # Update the '--start-batch' parameter in the job script with the determined batch number.
    sed -i "s/--start-batch [0-9]*/--start-batch $batch_num/" $code_dir/src/train_job.sh
}