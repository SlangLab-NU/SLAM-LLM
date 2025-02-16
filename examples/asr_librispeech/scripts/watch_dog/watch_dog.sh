#!/bin/bash

remote_user="zhang.jinda1"
remote_host="login.discovery.neu.edu"
remote_script_path="/work/van-speech-nlp/jindaznb/jslpnb/mllm_experiments/slam-llm/examples/asr_librispeech/scripts/slam_train.sh"
sleep_sec=3600

while true; do
	# SSH into the remote server and execute the check and submit script
	ssh -T ${remote_user}@${remote_host} "bash ${remote_script_path}"
	# Sleep for a specified amount of time before checking again
	sleep ${sleep_sec}
done