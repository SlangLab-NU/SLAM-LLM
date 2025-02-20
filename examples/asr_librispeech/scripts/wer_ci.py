import argparse
import glob
import os
import numpy as np
from jiwer import wer
from random import choices

def read_and_process_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    asr_lines = []
    phoneme_lines = []

    for i, line in enumerate(lines):
        parts = line.split('\t')
        if len(parts) > 1:
            if i % 2 == 0:  # ASR lines (even index rows)
                asr_lines.append(parts[1].strip())
            else:  # Phoneme recognition lines (odd index rows)
                phoneme_lines.append(parts[1].strip())

    return asr_lines, phoneme_lines

def remove_empty_lines(gt, pred):
    gt_non_empty, pred_non_empty = [], []
    
    for g, p in zip(gt, pred):
        if g.strip() and p.strip():  # Ensure lines are not empty or just spaces
            gt_non_empty.append(g.strip())
            pred_non_empty.append(p.strip())
    
    return gt_non_empty, pred_non_empty

def get_latest_file(file_pattern):
    files = glob.glob(file_pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def calculate_wer(gt, pred, label):
    if len(gt) == len(pred):
        score = wer(gt, pred)
        print(f"{label} WER: {score*100} %")
        return score
    else:
        print(f"Error: {label} line counts do not match (GT: {len(gt)}, PRED: {len(pred)}).")
        return None

def bootstrap_ci(gt, pred, n_bootstrap=1000, ci=95):
    wer_scores = []
    for _ in range(n_bootstrap):
        sampled_indices = choices(range(len(gt)), k=len(gt))
        sampled_gt = [gt[i] for i in sampled_indices]
        sampled_pred = [pred[i] for i in sampled_indices]
        wer_scores.append(wer(sampled_gt, sampled_pred))
    
    lower_bound = np.percentile(wer_scores, (100 - ci) / 2)
    upper_bound = np.percentile(wer_scores, 100 - (100 - ci) / 2)
    return lower_bound, upper_bound

def main(folder, separate, file):
    asr_wer = None  # Initialize asr_wer here
    
    if file:
        print(f"Using specified GT file: {file}")
        pred_file_path = file.replace('gt', 'pred')
    else:
        gt_file_pattern = os.path.join(folder, "decode_test_beam4_gt_*")
        file = get_latest_file(gt_file_pattern)
        if not file:
            print("Missing GT file.")
            return
        print(f"Using GT file: {file}")

        pred_file_pattern = os.path.join(folder, "decode_test_beam4_pred_*")
        pred_file_path = get_latest_file(pred_file_pattern)

        if not pred_file_path:
            print("Missing PRED file.")
            return
    
    print(f"Using PRED file: {pred_file_path}")
    
    gt_asr, gt_phoneme = read_and_process_file(file)
    pred_asr, pred_phoneme = read_and_process_file(pred_file_path)
    
    gt_asr, pred_asr = remove_empty_lines(gt_asr, pred_asr)
    gt_phoneme, pred_phoneme = remove_empty_lines(gt_phoneme, pred_phoneme)
    
    if separate:
        asr_wer = calculate_wer(gt_asr, pred_asr, "ASR")
        phoneme_wer = calculate_wer(gt_phoneme, pred_phoneme, "Phoneme")
    else:
        gt_combined = gt_asr + gt_phoneme
        pred_combined = pred_asr + pred_phoneme
        combined_wer = calculate_wer(gt_combined, pred_combined, "Combined")
        asr_wer = combined_wer
        
    # Calculate bootstrap CI for ASR WER
    if asr_wer is not None:
        ci_lower, ci_upper = bootstrap_ci(gt_asr, pred_asr)
        margin_of_error = (ci_upper - ci_lower) / 2
        # Convert WER and CI to percentages
        asr_wer_percent = asr_wer * 100
        margin_of_error_percent = margin_of_error * 100
        print(f"{asr_wer_percent:.2f} $\pm$ {margin_of_error_percent:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate WER from GT and PRED files, with CI for ASR")
    parser.add_argument('--folder', type=str, help="Path to the folder containing GT and PRED files")
    parser.add_argument('--separate', action='store_true', help="Calculate WER separately for ASR and phoneme lines (default: False)")
    parser.add_argument('--file', type=str, help="Path to the specific GT file (optional)")
    args = parser.parse_args()

    main(args.folder, args.separate, args.file)