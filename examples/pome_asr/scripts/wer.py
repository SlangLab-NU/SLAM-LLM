import argparse
import glob
import os
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

    return asr_lines, phoneme_lines


def remove_empty_lines(gt, pred):
    gt_non_empty, pred_non_empty = [], []

    for g, p in zip(gt, pred):
        if g.strip() and p.strip():  # Ensure lines are not empty or just spaces
            gt_non_empty.append(g.strip())  # Strip to remove unnecessary spaces
            pred_non_empty.append(p.strip())  # Strip to remove unnecessary spaces

    return gt_non_empty, pred_non_empty


def get_latest_file(file_pattern):
    files = glob.glob(file_pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def calculate_wer(gt, pred, label):
    if len(gt) == len(pred):
        score = wer(gt, pred)
        print(f"{label} WER: {score}")
    else:
        print(f"Error: {label} line counts do not match (GT: {len(gt)}, PRED: {len(pred)}).")

def print_combined_repeated_lines(repeated_asr, repeated_phoneme):
    """Combine and print repeated lines from both ASR and Phoneme."""
    combined_repeated_lines = repeated_asr.union(repeated_phoneme)
    print(f"\nFound {len(combined_repeated_lines)} repeated lines in total.")
    if combined_repeated_lines:
        print("Repeated lines are:")
        for line in combined_repeated_lines:
            print(f"- {line}")

def main(folder, separate, file):
    # Locate GT and PRED files
    if file:
        print(f"Using specified GT file: {file}")
        # Replace 'gt' with 'pred' in the specified GT file path to find the corresponding PRED file
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

    # Read files
    gt_asr, gt_phoneme = read_and_process_file(file)
    pred_asr, pred_phoneme = read_and_process_file(pred_file_path)

    # Remove empty lines
    gt_asr, pred_asr = remove_empty_lines(gt_asr, pred_asr)
    gt_phoneme, pred_phoneme = remove_empty_lines(gt_phoneme, pred_phoneme)

    if separate:
        # Calculate WER separately for ASR and phoneme
        calculate_wer(gt_asr, pred_asr, "ASR")
        calculate_wer(gt_phoneme, pred_phoneme, "Phoneme")
    else:
        # Combine ASR and phoneme lines for unified WER calculation
        gt_combined = gt_asr + gt_phoneme
        pred_combined = pred_asr + pred_phoneme
        calculate_wer(gt_combined, pred_combined, "Combined")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate WER from GT and PRED files")
    parser.add_argument('--folder', type=str, help="Path to the folder containing GT and PRED files")
    parser.add_argument('--separate', action='store_true', help="Calculate WER separately for ASR and phoneme lines (default: False)")
    parser.add_argument('--file', type=str, help="Path to the specific GT file (optional)")
    args = parser.parse_args()

    # `args.separate` will be False by default and True only if --separate is specified
    main(args.folder, args.separate, args.file)
