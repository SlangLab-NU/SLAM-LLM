#!/bin/bash
#SBATCH -N 1
#SBATCH -c 12
#SBATCH -p gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=08:00:00
#SBATCH --output=/work/van-speech-nlp/jindaznb/jslpnb/log/%j.output
#SBATCH --error=/work/van-speech-nlp/jindaznb/jslpnb/log/%j.error

#export PYTHONPATH=/root/whisper:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1
export HYDRA_FULL_ERROR=1

module load anaconda3/3.7
module load ffmpeg/20190305
source activate /work/van-speech-nlp/jindaznb/slamenv/

identifier="ami_w2v2_phi2"
echo "Identifier: $identifier"

run_dir=/work/van-speech-nlp/jindaznb/jslpnb/mllm_expriments/slam-llm
cd $run_dir
code_dir=examples/asr_librispeech

speech_encoder_path=/dev/null
llm_path=${run_dir}/models/phi-2

output_dir=/work/van-speech-nlp/jindaznb/jslpnb/mllm_expriments/slam-llm/out/${identifier}

