from datasets import load_dataset, Dataset
from tqdm import tqdm
import os
import json

# Load train-100 dataset (100-hour subset of the clean training data)
librispeech_train_100 = load_dataset(
    "librispeech_asr", 
    "clean", 
    split="train.100", 
    cache_dir="/work/van-speech-nlp/cache", 
    trust_remote_code=True
)

# Load validation (dev-360) dataset
librispeech_val_360 = load_dataset(
    "librispeech_asr", 
    "clean", 
    split="validation", 
    cache_dir="/work/van-speech-nlp/cache", 
    trust_remote_code=True
)

# Define local save paths for train-100 and val-360
save_path_train_100 = "/work/van-speech-nlp/librispeech/librispeech_train_clean_100"
save_path_val_360 = "/work/van-speech-nlp/librispeech/librispeech_val_360"

# Save the datasets to local disk
# librispeech_train_100.save_to_disk(save_path_train_100)
# librispeech_val_360.save_to_disk(save_path_val_360)

# To load the datasets from disk later
loaded_librispeech_train_100 = Dataset.load_from_disk(save_path_train_100)
loaded_librispeech_val_360 = Dataset.load_from_disk(save_path_val_360)

# Process and save train-clean-100 dataset
data_train_100 = []
for example in tqdm(loaded_librispeech_train_100, desc="Processing train-clean-100 dataset"):
    text = example['text'].lower()
    speaker_id = example['speaker_id']
    chapter_id = example['chapter_id']
    example_id = example['id']

    new_file = f"/work/van-speech-nlp/librispeech/LibriSpeech/train-clean-100/{speaker_id}/{chapter_id}/{example_id}.flac"
    
    if os.path.exists(new_file):
        example['file'] = new_file
        jsonl_entry = {
            "key": example_id,
            "source": new_file,
            "target": text
        }
        data_train_100.append(jsonl_entry)

print(data_train_100[0])

# Write train-clean-100 dataset to JSONL
with open('loaded_librispeech_train_clean_100.jsonl', 'w') as f:
    for entry in data_train_100:
        f.write(json.dumps(entry) + '\n')

# Process and save val-360 dataset
data_val_360 = []
for example in tqdm(loaded_librispeech_val_360, desc="Processing val-360 dataset"):
    text = example['text'].lower()
    speaker_id = example['speaker_id']
    chapter_id = example['chapter_id']
    example_id = example['id']

    new_file = f"/work/van-speech-nlp/librispeech/LibriSpeech/dev-clean/{speaker_id}/{chapter_id}/{example_id}.flac"
    
    if os.path.exists(new_file):
        example['file'] = new_file
        jsonl_entry = {
            "key": example_id,
            "source": new_file,
            "target": text
        }
        data_val_360.append(jsonl_entry)

# Write val-360 dataset to JSONL
with open('loaded_librispeech_val_360.jsonl', 'w') as f:
    for entry in data_val_360:
        f.write(json.dumps(entry) + '\n')