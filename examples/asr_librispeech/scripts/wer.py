import argparse
import glob
import os
from jiwer import wer

# Function to read and process the file
def read_and_process_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    processed_lines = []
    for line in lines:
        parts = line.split('\t')
        if len(parts) > 1:
            processed_lines.append(parts[1].strip())
    
    return processed_lines

def remove_empty_lines(gt, pred):
    gt_non_empty, pred_non_empty = [], []
    
    for g, p in zip(gt, pred):
        if g and p:  # Only include non-empty lines
            gt_non_empty.append(g)
            pred_non_empty.append(p)
    
    return gt_non_empty, pred_non_empty

def filter_repeated_words(gt, pred, max_ngram=3):
    """
    Filters lines with repeated n-grams from predictions.

    Args:
        gt (list): Ground truth lines.
        pred (list): Prediction lines.
        max_ngram (int): Maximum n-gram size to check for repetitions.

    Returns:
        filtered_gt (list): Filtered ground truth lines.
        filtered_pred (list): Filtered prediction lines.
        repeated_lines (set): Set of removed lines from predictions.
    """
    filtered_gt, filtered_pred = [], []
    repeated_lines = set()
    
    def has_repeated_ngram(words, max_ngram):
        """Check if any n-gram is repeated at least three times consecutively in the words."""
        for n in range(1, max_ngram + 1):  # Check from 1-gram to max_ngram
            for i in range(len(words) - 2 * n):  # Ensure space for three consecutive n-grams
                ngram = words[i:i + n]
                if words[i + n:i + 2 * n] == ngram and words[i + 2 * n:i + 3 * n] == ngram:
                    return True
        return False

    for g, p in zip(gt, pred):
        words = p.split()
        if has_repeated_ngram(words, max_ngram):
            repeated_lines.add(p)
        else:
            filtered_gt.append(g)
            filtered_pred.append(p)

    return filtered_gt, filtered_pred, repeated_lines



def get_latest_file(file_pattern):
    files = glob.glob(file_pattern)
    if not files:
        return None
    # Select the most recent file based on the last modified timestamp
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def main(folder):
    # Print out the folder path being used
    print(f"Using folder: {folder}")

    # Locate the most recent GT file
    gt_file_pattern = os.path.join(folder, "decode_test_beam4_gt_*")
    gt_file_path = get_latest_file(gt_file_pattern)
    if not gt_file_path:
        print(f"No GT file matching pattern '{gt_file_pattern}' found.")
        return
    print(f"Using GT file: {gt_file_path}")
    gt = read_and_process_file(gt_file_path)

    # Locate the most recent PRED file
    pred_file_pattern = os.path.join(folder, "decode_test_beam4_pred_*")
    pred_file_path = get_latest_file(pred_file_pattern)
    if not pred_file_path:
        print(f"No PRED file matching pattern '{pred_file_pattern}' found.")
        return
    print(f"Using PRED file: {pred_file_path}")
    pred = read_and_process_file(pred_file_path)

    # Remove empty lines from GT and PRED
    gt, pred = remove_empty_lines(gt, pred)

    # Calculate WER before filtering repeated words
    if len(gt) == len(pred):
        initial_wer_score = wer(gt, pred)
        print(f"Initial Word Error Rate (WER) before filtering: {initial_wer_score}")
    else:
        print("Error: The number of lines in the GT and PRED files do not match before filtering.")
        return

    # Filter repeated words in PRED and adjust GT accordingly
    original_pred_len = len(pred)
    gt, pred, repeated_lines = filter_repeated_words(gt, pred)
    filtered_count = original_pred_len - len(pred)

    # Print lengths of GT and PRED lists to ensure they are the same
    print(f"Number of GT lines after filtering: {len(gt)}")
    print(f"Number of original PRED lines: {original_pred_len}")
    print(f"Number of filtered repeated lines: {filtered_count} out of {original_pred_len}")
    
    # Print repeated lines
    if repeated_lines:
        print("Lines with repeated words in PRED file:")
        for line in repeated_lines:
            print(f"- {line}")

    # Calculate WER after filtering repeated words
    if len(gt) == len(pred):
        filtered_wer_score = wer(gt, pred)
        print(f"Filtered Word Error Rate (WER) after removing repeated lines: {filtered_wer_score}")
    else:
        print("Error: The number of lines in the GT and filtered PRED files do not match after filtering.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate WER from GT and PRED files")
    parser.add_argument('--folder', type=str, required=True, help="Path to the folder containing GT and PRED files")
    args = parser.parse_args()
    
    main(args.folder)
