import argparse
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

def check_empty_entries(data):
    empty_lines = []
    
    # Iterate through each line in the data
    for index, line in enumerate(data):
        # Split the line by tab characters
        parts = line.split("\t")
        
        # Ensure there are at least two parts (ID and transcription) and check if transcription is empty
        if len(parts) < 2 or not parts[1].strip():
            empty_lines.append(index + 1)  # Append 1-based index of the line
    
    return empty_lines

def main(folder, pred_file_name):
    # Read and process GT file
    gt_file_path = f"{folder}/decode_test_beam4_gt"
    gt_asr, gt_phoneme, gt_lines = read_and_process_file(gt_file_path)

    # Read and process PRED file
    pred_file_path = f"{folder}/{pred_file_name}"
    pred_asr, pred_phoneme, pred_lines = read_and_process_file(pred_file_path)

    # Print lengths of GT and PRED lists to ensure they are the same
    print(f"Number of GT ASR lines: {len(gt_asr)}")
    print(f"Number of PRED ASR lines: {len(pred_asr)}")
    print(f"Number of GT Phoneme lines: {len(gt_phoneme)}")
    print(f"Number of PRED Phoneme lines: {len(pred_phoneme)}")

    # Print first 2 ASR lines for verification
    print("\nFirst 2 ASR lines from GT file:")
    for line in gt_asr[:2]:
        print(line)

    print("\nFirst 2 ASR lines from PRED file:")
    for line in pred_asr[:2]:
        print(line)

    # Print first 2 phoneme lines for verification
    print("\nFirst 2 Phoneme lines from GT file:")
    for line in gt_phoneme[:2]:
        print(line)

    print("\nFirst 2 Phoneme lines from PRED file:")
    for line in pred_phoneme[:2]:
        print(line)

    # Calculate WER for ASR
    if len(gt_asr) == len(pred_asr):
        wer_asr_score = wer(gt_asr, pred_asr)
        print(f"Word Error Rate (WER) for ASR: {wer_asr_score}")
    else:
        print("Error: The number of lines in the GT and PRED ASR files do not match.")

    # Remove empty phoneme entries from both GT and PRED phoneme lists
    filtered_gt_phoneme, filtered_pred_phoneme = filter_empty_phoneme_entries(gt_phoneme, pred_phoneme)

    # Calculate PER for phoneme recognition
    if len(filtered_gt_phoneme) == len(filtered_pred_phoneme):
        wer_phoneme_score = wer(filtered_gt_phoneme, filtered_pred_phoneme)
        print(f"Phoneme Error Rate (PER) for phoneme recognition: {wer_phoneme_score}")
    else:
        print("Error: The number of lines in the GT and PRED phoneme files do not match after filtering.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate WER for ASR and PER for phoneme recognition from GT and PRED files")
    parser.add_argument('--folder', type=str, required=True, help="Path to the folder containing GT and PRED files")
    parser.add_argument('--pred_file_name', type=str, default='decode_test_beam4_pred', help="Name of the PRED file")
    args = parser.parse_args()
    
    main(args.folder, args.pred_file_name)