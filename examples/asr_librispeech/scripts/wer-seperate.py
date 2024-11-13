import argparse
import os
import glob
from jiwer import wer

# Function to read and process the file
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

    return asr_lines, phoneme_lines, lines  # Return original lines for verification

# Function to remove empty phoneme lines from both GT and PRED
def filter_empty_phoneme_entries(gt_phoneme, pred_phoneme):
    filtered_gt_phoneme = []
    filtered_pred_phoneme = []
    
    for gt, pred in zip(gt_phoneme, pred_phoneme):
        if gt.strip() and pred.strip():  # Ensure neither is empty
            filtered_gt_phoneme.append(gt)
            filtered_pred_phoneme.append(pred)
    
    return filtered_gt_phoneme, filtered_pred_phoneme

# Function to filter repeated words for five consecutive repetitions
def filter_repeated_words(gt, pred):
    filtered_gt, filtered_pred = [], []
    repeated_lines = set()
    
    for g, p in zip(gt, pred):
        words = p.split()
        repeated_word = None
        
        # Check if there are five consecutive occurrences of the same word
        for i in range(len(words) - 4):  # Adjusted to `-4` to check up to five consecutive words
            if words[i] == words[i + 1] == words[i + 2] == words[i + 3] == words[i + 4]:
                repeated_word = words[i]
                break

        # If no repeated words are found, add to filtered lists
        if repeated_word:
            repeated_lines.add(p)
        else:
            filtered_gt.append(g)
            filtered_pred.append(p)
    
    return filtered_gt, filtered_pred, repeated_lines

def main(folder):
    # Locate GT file with flexible naming using glob
    gt_file_pattern = os.path.join(folder, "decode_test_beam4_*_gt")
    gt_files = glob.glob(gt_file_pattern)

    if not gt_files:
        print("No GT file found.")
        return
    gt_file_path = gt_files[0]  # Assuming only one GT file is expected
    print(f"Using GT file: {gt_file_path}")

    # Read and process GT file
    gt_asr, gt_phoneme, gt_lines = read_and_process_file(gt_file_path)

    # Locate PRED file with flexible naming using glob
    pred_file_pattern = os.path.join(folder, "decode_test_beam4_*_pred")
    pred_files = glob.glob(pred_file_pattern)

    if not pred_files:
        print("No PRED file found.")
        return
    pred_file_path = pred_files[0]  # Assuming only one PRED file is expected
    print(f"Using PRED file: {pred_file_path}")

    # Read and process PRED file
    pred_asr, pred_phoneme, pred_lines = read_and_process_file(pred_file_path)

    # Initial WER for ASR before filtering
    if len(gt_asr) == len(pred_asr):
        initial_wer_asr_score = wer(gt_asr, pred_asr)
        print(f"Initial WER for ASR before filtering: {initial_wer_asr_score}")
    else:
        print("Error: The number of lines in the GT and PRED ASR files do not match before filtering.")
        return

    # Filter repeated words for ASR lines and adjust GT accordingly
    original_asr_len = len(pred_asr)
    gt_asr, pred_asr, repeated_asr_lines = filter_repeated_words(gt_asr, pred_asr)
    filtered_asr_count = original_asr_len - len(pred_asr)

    # Print lengths after filtering
    print(f"Number of filtered repeated ASR lines: {filtered_asr_count} out of {original_asr_len}")

    # Filtered WER for ASR
    if len(gt_asr) == len(pred_asr):
        filtered_wer_asr_score = wer(gt_asr, pred_asr)
        print(f"Filtered WER for ASR after removing repeated lines: {filtered_wer_asr_score}")
    else:
        print("Error: The number of lines in the GT and filtered PRED ASR files do not match after filtering.")
        return

    # Remove empty phoneme entries from both GT and PRED phoneme lists
    filtered_gt_phoneme, filtered_pred_phoneme = filter_empty_phoneme_entries(gt_phoneme, pred_phoneme)

    # Check if the filtered lists are not empty before calculating PER
    if not filtered_gt_phoneme or not filtered_pred_phoneme:
        print("Error: One of the filtered phoneme lists is empty, cannot calculate PER.")
        return

    # Initial PER before filtering repeated phoneme lines
    if len(filtered_gt_phoneme) == len(filtered_pred_phoneme):
        initial_per_phoneme_score = wer(filtered_gt_phoneme, filtered_pred_phoneme)
        print(f"Initial PER for phoneme recognition before filtering: {initial_per_phoneme_score}")
    else:
        print("Error: The number of lines in the GT and PRED phoneme files do not match before filtering.")
        return

    # Filter repeated words for phoneme lines and adjust GT accordingly
    original_phoneme_len = len(filtered_pred_phoneme)
    filtered_gt_phoneme, filtered_pred_phoneme, repeated_phoneme_lines = filter_repeated_words(filtered_gt_phoneme, filtered_pred_phoneme)
    filtered_phoneme_count = original_phoneme_len - len(filtered_pred_phoneme)

    # Print lengths after filtering
    print(f"Number of filtered repeated phoneme lines: {filtered_phoneme_count} out of {original_phoneme_len}")

    # Filtered PER for phoneme recognition
    if len(filtered_gt_phoneme) == len(filtered_pred_phoneme):
        filtered_per_phoneme_score = wer(filtered_gt_phoneme, filtered_pred_phoneme)
        print(f"Filtered PER for phoneme recognition after removing repeated lines: {filtered_per_phoneme_score}")
    else:
        print("Error: The number of lines in the GT and filtered PRED phoneme files do not match after filtering.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate WER for ASR and PER for phoneme recognition from GT and PRED files")
    parser.add_argument('--folder', type=str, required=True, help="Path to the folder containing GT and PRED files")
    args = parser.parse_args()
    
    main(args.folder)
