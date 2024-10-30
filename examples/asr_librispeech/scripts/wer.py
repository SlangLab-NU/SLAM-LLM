import argparse
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

def main(folder, pred_file_name):
    # Read and process GT file
    gt_file_path = f"{folder}/decode_test_beam4_gt"
    gt = read_and_process_file(gt_file_path)

    # Read and process PRED file
    pred_file_path = f"{folder}/decode_test_beam4_pred"
    pred = read_and_process_file(pred_file_path)

    # Remove empty lines from GT and PRED
    gt, pred = remove_empty_lines(gt, pred)

    # Print lengths of GT and PRED lists to ensure they are the same
    print(f"Number of GT lines: {len(gt)}")
    print(f"Number of PRED lines: {len(pred)}")

    # Calculate WER
    if len(gt) == len(pred):
        wer_score = wer(gt, pred)
        print(f"Word Error Rate (WER): {wer_score}")
    else:
        print("Error: The number of lines in the GT and PRED files do not match.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate WER from GT and PRED files")
    parser.add_argument('--folder', type=str, required=True, help="Path to the folder containing GT and PRED files")
    parser.add_argument('--pred_file_name', type=str, default='decode_test_beam4_pred', help="Name of the PRED file")
    args = parser.parse_args()
    
    main(args.folder, args.pred_file_name)