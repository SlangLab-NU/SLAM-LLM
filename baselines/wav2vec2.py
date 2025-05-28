# Import libraries
import sys
import os
import argparse
import re
import json
import torch
import logging
import pandas as pd
import numpy as np

from dotenv import load_dotenv
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict, Audio
from dataclasses import dataclass
from typing import Dict, List, Union
from tqdm import tqdm
from datetime import datetime
import jiwer

# Load environment variables from .env file
load_dotenv()

# Explicitly set Hugging Face cache environment variables
os.environ["HF_HOME"] = os.environ.get("HF_HOME", "/work/van-speech-nlp/temp/huggingface_cache")
os.environ["TRANSFORMERS_CACHE"] = os.environ.get("TRANSFORMERS_CACHE", "/work/van-speech-nlp/temp/huggingface_cache/models")
os.environ["HF_DATASETS_CACHE"] = os.environ.get("HF_DATASETS_CACHE", "/work/van-speech-nlp/temp/huggingface_cache/datasets")

# Create cache directories if they don't exist
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)

# Parse command line arguments - only keep essential ones
parser = argparse.ArgumentParser(description='Fine-tune Wav2Vec2 model for ASR')
parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset folder (e.g., aphasia)')
parser.add_argument('--pretrained_model', type=str, 
                    default='facebook/wav2vec2-large-xlsr-53',
                    help='Pretrained model name (default: facebook/wav2vec2-large-xlsr-53)')
parser.add_argument('--debug', action='store_true',
                    help='Enable debug mode (default: False)')
parser.add_argument('--repo_suffix', type=str,
                    default='', help='Repository suffix')
parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint if available (default: False)')
args = parser.parse_args()

# Load training configuration from JSON
with open('training_args.json') as training_args_file:
    training_args_dict = json.load(training_args_file)

# Extract model name from pretrained_model_name
model_name = args.pretrained_model.split('/')[-1]

output_path = 'output'
# Path to save model / checkpoints
model_local_path = output_path + f'/model/{model_name}_{args.dataset}_asr'

# Model to be fine-tuned
# pretrained_model_name = "facebook/wav2vec2-large-xlsr-53"

if not os.path.exists(output_path + '/logs'):
    os.makedirs(output_path + '/logs')

log_dir = f'{output_path}/logs/{model_name}_{args.dataset}_asr'

# Create the results directory for the current speaker, if it does not exist
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file_name = 'train_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.log'
log_file_path = log_dir + '/' + log_file_name

logging.basicConfig(
    filename=log_file_path,
    filemode='a',
    format='%(asctime)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
    level=logging.INFO
)

# Log to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(console_handler)




'''
--------------------------------------------------------------------------------
Read the dataset from JSONL files
--------------------------------------------------------------------------------
'''
# Load the JSONL datasets
dataset_files = {
    'train': f'../data/{args.dataset}/train.jsonl',
    'validation': f'../data/{args.dataset}/validation.jsonl',
    'test': f'../data/{args.dataset}/test.jsonl'
}

# Log dataset information
logging.info(f"Loading dataset from {args.dataset} folder")
logging.info(f"Dataset files: {dataset_files}")

dataset = load_dataset('json', data_files=dataset_files)

# Check if the following fields exist in the dataset ['source', 'target']
expected_fields = ['source', 'target']
not_found_fields = []
for field in expected_fields:
    if field not in dataset['train'].column_names:
        not_found_fields.append(field)

if len(not_found_fields) > 0:
    logging.error(
        "The following fields are not found in the dataset:" + " [" + ", ".join(not_found_fields) + "]")
    sys.exit(1)

'''
--------------------------------------------------------------------------------
Use GPU if available
--------------------------------------------------------------------------------
'''
if torch.cuda.is_available():
    device = torch.device("cuda")
    logging.info("Using GPU: " + torch.cuda.get_device_name(0) + '\n')
else:
    device = torch.device("cpu")
    logging.info("Using CPU\n")



'''
--------------------------------------------------------------------------------
Build Processor with Tokenizer and Feature Extractor
--------------------------------------------------------------------------------
'''
# Remove special characters from the text
chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\"\%\'\"\`\0-9]'

def remove_special_characters(batch):
    batch['text'] = re.sub(chars_to_ignore_regex,
                           ' ', batch['target']).lower()
    return batch

dataset = dataset.map(remove_special_characters)

# Create a dictionary of tokenizer vocabularies
vocab_list = []
for split in dataset.values():
    for text in split['text']:
        text = text.replace(' ', '|')
        vocab_list.extend(text)

vocab_dict = {}
vocab_dict['[PAD]'] = 0
vocab_dict['<s>'] = 1
vocab_dict['</s>'] = 2
vocab_dict['[UNK]'] = 3
vocab_list = sorted(list(set(vocab_list)))
vocab_dict.update({v: k + len(vocab_dict)
                  for k, v in enumerate(vocab_list)})

logging.info("Vocab Dictionary:")
logging.info(str(vocab_dict) + '\n')

# Create a directory to store the vocab.json file
if not os.path.exists(output_path + '/vocab'):
    os.makedirs(output_path + '/vocab')

vocab_file_name = f'{model_name}_{args.dataset}_asr_vocab.json'
vocab_file_path = output_path + '/vocab/' + vocab_file_name

# Save the vocab.json file
with open(vocab_file_path, 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)

# Build the tokenizer
tokenizer = Wav2Vec2CTCTokenizer(
    vocab_file_path, unk_token='[UNK]', pad_token='[PAD]')

# Build the feature extractor
sampling_rate = 16000
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1, sampling_rate=sampling_rate, padding_value=0.0, return_attention_mask=True)

# Build the processor
processor = Wav2Vec2Processor(
    feature_extractor=feature_extractor, tokenizer=tokenizer)

'''
--------------------------------------------------------------------------------
Preprocess the dataset
- Load and resample the audio data
- Extract values from the loaded audio file
- Encode the transcriptions to label ids
--------------------------------------------------------------------------------
'''
def prepare_dataset(batch):
    # Load audio data into batch
    audio = batch['audio']

    # Extract values
    batch["input_values"] = processor(
        audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])

    # Encode to label ids
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids

    return batch

import multiprocessing
cpu_cnt=multiprocessing.cpu_count()
print(cpu_cnt)
# Update audio paths and load audio data
dataset = dataset.cast_column("source", Audio(sampling_rate=sampling_rate))
dataset = dataset.rename_column("source", "audio")
dataset = dataset.map(prepare_dataset, remove_columns=[
                                  'key', 'audio', 'target', 'text'], num_proc=cpu_cnt)
# Print first few rows after processing
print(dataset['train'].select(range(3)))

for split in dataset:
    logging.info(f"{split}: {len(dataset[split])} samples")
logging.info("\n")

# Remove the "input_length" column as it's no longer needed
dataset = dataset.remove_columns(["input_length"])

'''
--------------------------------------------------------------------------------
Define a DataCollator:
wave2vec2 has a much larger input length as compared to the output length. For
the input size, it is efficient to pad training batches to the longest sample
in the batch (not overall sample)
--------------------------------------------------------------------------------
'''
# Define the data collator
@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
        The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
        Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
        among:
        * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
            sequence if provided).
        * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
            maximum acceptable input length for the model if that argument is not provided.
        * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
            different lengths).
    """
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]}
                          for feature in features]
        label_features = [{"input_ids": feature["labels"]}
                          for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels

        return batch

data_collator = DataCollatorCTCWithPadding(
    processor=processor, padding=True)

'''
--------------------------------------------------------------------------------
Define the Evaluation Metrics
--------------------------------------------------------------------------------
'''
# Replace Hugging Face evaluate with jiwer
def compute_metrics(pred):
    """
        Compute Word Error Rate (WER) for the model predictions.

        Parameters:
        pred (transformers.file_utils.ModelOutput): Model predictions.

        Returns:
        dict: A dictionary containing the computed metrics.
    """
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -
                   100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    
    # Use jiwer instead of Hugging Face evaluate
    wer = jiwer.wer(label_str, pred_str)

    logging.info("Current Word Error Rate: " + str(wer))

    return {"wer": wer}

'''
--------------------------------------------------------------------------------
Load the model
--------------------------------------------------------------------------------
'''
# Load the model
model = Wav2Vec2ForCTC.from_pretrained(
    args.pretrained_model,
    ctc_loss_reduction="mean",
    vocab_size=len(processor.tokenizer),
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    mask_time_length=2,
    layerdrop=0.1,
    use_auth_token="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"
)

# Freeze the feature extractor
# (parameters of pre-trained part of the model won't be updated during training)
model.freeze_feature_encoder()

# Release unoccupied cache memory
torch.cuda.empty_cache()

'''
--------------------------------------------------------------------------------
Define the Training Arguments
--------------------------------------------------------------------------------
'''

# Load the training arguments from training_args.json
with open('training_args.json') as training_args_file:
    training_args_dict = json.load(training_args_file)

# # Replace the default values with the values from the command line arguments
training_args_dict['learning_rate'] = training_args_dict['learning_rate']
training_args_dict['per_device_train_batch_size'] = training_args_dict['per_device_train_batch_size']
training_args_dict['per_device_eval_batch_size'] = training_args_dict['per_device_eval_batch_size']
training_args_dict['seed'] = training_args_dict['seed']
# Calculate gradient_accumulation_steps as 16/per_device_train_batch_size
training_args_dict['gradient_accumulation_steps'] = 16 // training_args_dict['per_device_train_batch_size']
training_args_dict['optim'] = training_args_dict['optim']
training_args_dict['lr_scheduler_type'] = training_args_dict['lr_scheduler_type']
training_args_dict['num_train_epochs'] = training_args_dict['num_train_epochs']

# Create the model directory, if it does not exist
if not os.path.exists(output_path + '/model'):
    os.makedirs(output_path + '/model')

# Define the training arguments
training_args = TrainingArguments(
    output_dir=model_local_path,
    metric_for_best_model="wer",  # Use WER to determine best model
    greater_is_better=False,  # Lower WER is better
    load_best_model_at_end=True,  # Load the best model at the end of training
    **training_args_dict
)

'''
--------------------------------------------------------------------------------
Print Training Configuration
--------------------------------------------------------------------------------
'''
logging.info("\n=== Training Configuration ===")
logging.info("Model Configuration:")
logging.info(f"- Pretrained Model: {args.pretrained_model}")
logging.info(f"- Dataset: {args.dataset}")
logging.info(f"- Output Path: {model_local_path}")

logging.info("\nTraining Parameters:")
for key, value in training_args_dict.items():
    logging.info(f"- {key}: {value}")

logging.info("\nSystem Information:")
logging.info(f"- Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
if torch.cuda.is_available():
    logging.info(f"- GPU: {torch.cuda.get_device_name(0)}")
    logging.info(f"- Available GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
logging.info("===============================\n")

'''
--------------------------------------------------------------------------------
Define the Trainer
--------------------------------------------------------------------------------
'''
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=processor.feature_extractor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.create_model_card(
    language="en",
    tags=["audio", "speech", model_name, args.dataset],
    model_name=f"{model_name}_{args.dataset}_asr",
    finetuned_from=args.pretrained_model,
    tasks=["automatic-speech-recognition"],
    dataset=args.dataset,
    dataset_tags=[args.dataset],
    dataset_args=f"{args.dataset.capitalize()} Dataset",
    # dataset_url="",
)

# Add debug logging before training
logging.info("Dataset sizes:")
logging.info(f"Train: {len(dataset['train'])} samples")
logging.info(f"Validation: {len(dataset['validation'])} samples")
logging.info(f"Test: {len(dataset['test'])} samples")

# Print and log detailed training info before training
logging.info(f"Model: {args.pretrained_model}")
logging.info(f"Dataset: {args.dataset}")
logging.info(f"Train size: {len(dataset['train'])}")
logging.info(f"Validation size: {len(dataset['validation'])}")
logging.info(f"Test size: {len(dataset['test'])}")
logging.info(f"Training arguments: {training_args_dict}")
print(f"Model: {args.pretrained_model}")
print(f"Dataset: {args.dataset}")
print(f"Train size: {len(dataset['train'])}")
print(f"Validation size: {len(dataset['validation'])}")
print(f"Test size: {len(dataset['test'])}")
print(f"Training arguments: {training_args_dict}")

# Print a few samples after preprocessing for debugging
print("\n==== Sampled Preprocessed Training Data ====")
for i in range(min(3, len(dataset['train']))):
    sample = dataset['train'][i]
    print(f"Sample {i} input_values shape: {np.array(sample['input_values']).shape}")
    print(f"Sample {i} labels: {sample['labels']}")
    print(f"Sample {i} label tokens: {[processor.tokenizer.decode([x]) for x in sample['labels'] if x != -100]}")
    logging.info(f"Sample {i} input_values shape: {np.array(sample['input_values']).shape}")
    logging.info(f"Sample {i} labels: {sample['labels']}")
    logging.info(f"Sample {i} label tokens: {[processor.tokenizer.decode([x]) for x in sample['labels'] if x != -100]}")
print("===========================================\n")

logging.info("\nStarting trainer.train()...")

'''
--------------------------------------------------------------------------------
Start Training
--------------------------------------------------------------------------------
'''
logging.info("Start Training")
logging.info("Training Arguments:")
training_arg_log_dict = {"Training Epochs": training_args_dict['num_train_epochs'],
                         "Training Batch Size": training_args_dict['per_device_train_batch_size'],
                         "Evaluation Batch Size": training_args_dict['per_device_eval_batch_size'],
                         "Learning Rate": training_args_dict['learning_rate'],
                         "Weight Decay": training_args_dict['weight_decay']}
logging.info(str(training_arg_log_dict))

train_start_time = datetime.now()

# Train from scratch if there is no checkpoint in the repository
# Check if checkpoint-* directories exist in the repository
checkpoint_files = [f for f in os.listdir(model_local_path) if f.startswith(
    'checkpoint-') and os.path.isdir(os.path.join(model_local_path, f))]
if args.resume and len(checkpoint_files) > 0:
    logging.info(
        f"Checkpoint found in the repository. Checkpoint files found: {checkpoint_files}")
    resume_from_checkpoint = f"{model_local_path}/{checkpoint_files[-1]}"
    logging.info(f"Resuming from checkpoint: {resume_from_checkpoint}\n")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
else:
    if args.resume:
        logging.info("No checkpoint found, starting training from scratch.")
    else:
        logging.info("--resume not set, starting training from scratch.")
    trainer.train()

train_end_time = datetime.now()

logging.info("Training completed in " +
             str(train_end_time - train_start_time) + '\n')

logging.info("Training Log Metrics:")
for history in trainer.state.log_history:
    logging.info(str(history))

'''
--------------------------------------------------------------------------------
Predict and evaluate on the test set
--------------------------------------------------------------------------------
'''
logging.info("Starting evaluation on test set")

# Load the best model for evaluation
best_model_path = trainer.state.best_model_checkpoint
if best_model_path:
    logging.info(f"Loading best model from {best_model_path}")
    model = Wav2Vec2ForCTC.from_pretrained(best_model_path)
else:
    logging.info("No best model checkpoint found, using current model")

# Move model to GPU if available
if torch.cuda.is_available():
    model = model.to("cuda")
    logging.info("Model moved to GPU for evaluation")

def predict_dataset(dataset):
    '''
    Predict on the dataset

    Parameters:
    dataset (datasets.Dataset): Dataset to predict on

    Returns:
    predictions (list): List of predictions
    references (list): List of references
    '''

    predictions = []
    references = []

    for i in tqdm(range(dataset.num_rows)):
        inputs = processor(
            dataset[i]["input_values"], sampling_rate=sampling_rate, return_tensors="pt", padding=True)

        # Move input to GPU
        if torch.cuda.is_available():
            inputs = {key: val.to("cuda") for key, val in inputs.items()}

        # Predict
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        prediction = processor.batch_decode(predicted_ids)[0].lower()

        # Decode the reference from label ids
        reference_ids = dataset[i]["labels"]
        reference = processor.decode(reference_ids, group_tokens=False).lower()

        predictions.append(prediction)
        references.append(reference)

    return predictions, references

# Predict on test set
logging.info("Generating predictions for test set")
test_predictions, test_references = predict_dataset(dataset["test"])

# Calculate WER for the test set
test_wer = jiwer.wer(test_references, test_predictions)
logging.info(f"Test set Word Error Rate: {test_wer}")

# Save predictions and references to a file for further analysis
predictions_dir = f"{output_path}/predictions"
if not os.path.exists(predictions_dir):
    os.makedirs(predictions_dir)

predictions_file = f"{predictions_dir}/{model_name}_{args.dataset}_test_predictions.json"
with open(predictions_file, 'w') as f:
    json.dump({
        'predictions': test_predictions,
        'references': test_references,
        'wer': test_wer
    }, f, indent=2)
logging.info(f"Predictions saved to {predictions_file}")

'''
--------------------------------------------------------------------------------
End of Script
--------------------------------------------------------------------------------
'''

logging.info("End of Script")
logging.info("--------------------------------------------\n")
