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
    # Print out the folder path being used
    print(f"Using folder: {folder}")

    # Locate GT file with flexible naming using glob
    gt_file_pattern = os.path.join(folder, "decode_test_beam4_*_gt")
    gt_files = glob.glob(gt_file_pattern)
    
    # Check if any matching GT file is found
    if not gt_files:
        print(f"No GT file matching pattern '{gt_file_pattern}' found.")
        return
    elif len(gt_files) > 1:
        print(f"Multiple GT files found. Using the first one: {gt_files[0]}")
    
    gt_file_path = gt_files[0]  # Select the first matched file
    gt = read_and_process_file(gt_file_path)

    # Locate PRED file with flexible naming using glob
    pred_file_pattern = os.path.join(folder, "decode_test_beam4_*_pred")
    pred_files = glob.glob(pred_file_pattern)
    pred_file_path = pred_files[0]  # Select the first matched file
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
