import os
import json
import re
from tqdm import tqdm
from g2p import make_g2p

# Initialize G2P converter for English ARPAbet
transducer = make_g2p('eng', 'eng-arpabet')

def clean_phonemes(phonemes):
    """
    Cleans the phonemes string by:
    1. Removing special characters like "'" and '"'.
    2. Removing trailing and multiple spaces.
    """
    # Remove special characters
    cleaned = re.sub(r"[^\w\s]", "", phonemes)
    # Remove extra spaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def get_phonemes(sentence):
    """Convert sentence to phonemes using G2P and clean the result."""
    phonemes_list = [transducer(word).output_string for word in re.findall(r'\S+', sentence)]
    phonemes = " ".join(phonemes_list)
    return clean_phonemes(phonemes)  # Clean the phoneme transcript

def process_json_files(input_folder):
    """Process all JSON files in the input folder and save the corresponding updated JSON files to the output folder."""
    # Dynamically create the output folder name by appending '_phoneme'
    output_folder = f"{input_folder.rstrip(os.sep)}_phoneme"
    
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all files in the input folder
    for file_name in tqdm(os.listdir(input_folder), desc="Processing Files"):
        if file_name.endswith('.jsonl'):  # Only process .jsonl files
            input_file_path = os.path.join(input_folder, file_name)
            output_file_path = os.path.join(output_folder, file_name)

            # Open the input JSON file
            with open(input_file_path, 'r', encoding='utf-8') as file:
                data = file.readlines()

            # Initialize list to store processed data
            updated_data = []

            # Iterate through each JSON line, process and update with phoneme transcript
            for line in tqdm(data, desc=f"Processing {file_name}"):
                data_dict = json.loads(line.strip())  # Load the JSON object

                # Update the 'target' field with the phoneme transcription
                target_sentence = data_dict.get('target', '')  # Get the target (transcription) sentence
                if target_sentence:
                    phoneme_transcript = get_phonemes(target_sentence)  # Get phoneme transcript
                    data_dict['target'] = phoneme_transcript  # Update the 'target' field

                # Update the 'prompt' field
                data_dict['prompt'] = (
                    "Transcribe speech to Phonemes. Output the transcription directly without redundant content. "
                    "Ensure that the output is not duplicated."
                )

                updated_data.append(data_dict)  # Store the updated object

            # Save the updated data back to the corresponding output JSON file
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                for updated_record in updated_data:
                    json.dump(updated_record, output_file)
                    output_file.write('\n')

# Define input folder
input_folder = "/work/van-speech-nlp/jindaznb/jslpnb/mllm_experiments/slam-llm/data/aphasia"  # Update this to your input folder path

# Process all JSON files
process_json_files(input_folder)

# %%



