#!/bin/bash -l
#SBATCH -N 1                
#SBATCH -c 12               
#SBATCH -p short              
#SBATCH --mem=16GB          
#SBATCH --time=02:00:00
#SBATCH --output=log/%j.output
#SBATCH --error=log/%j.error

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1
export HYDRA_FULL_ERROR=1

module purge
module load discovery
module load python/3.8.1 
module load anaconda3/3.7 
module load ffmpeg/20190305 
source activate /work/van-speech-nlp/jindaznb/slamenv/
which python


# python wer_ci.py --separate \
# 	--folder /work/van-speech-nlp/jindaznb/jslpnb/mllm_experiments/slam-llm/out/ami_phoneme_separate/wavlm_llama32_1b_linear_peft


# python wer_ci.py --separate \
# 	--folder /work/van-speech-nlp/jindaznb/jslpnb/mllm_experiments/slam-llm/out/aphasia_phoneme_separate/wavlm_llama32_1b_linear_peft_seed_42


# python wer_ci.py --separate \
# 	--folder /work/van-speech-nlp/jindaznb/jslpnb/mllm_experiments/slam-llm/out/librispeech-100_phoneme_separate/wavlm_llama32_1b_linear_peft


python wer_ci.py --separate \
	--folder /work/van-speech-nlp/jindaznb/jslpnb/mllm_experiments/slam-llm/out/librispeech-100_phoneme_separate/wavlm_llama32_1b_linear_peft/decode_test_beam4_gt_20250209_225046