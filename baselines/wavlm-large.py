import os
import json
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    WavLMConfig,
    WavLMForCTC,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2CTCTokenizer,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup
)
import soundfile as sf
import wandb
from tqdm.auto import tqdm
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = "microsoft/wavlm-large"
    cache_dir: Optional[str] = None
    use_auth_token: bool = False

@dataclass
class DataTrainingArguments:
    train_file: str = "../data/ami/train.jsonl"
    validation_file: str = "../data/ami/validation.jsonl"
    test_file: str = "../data/ami/test.jsonl"
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    max_test_samples: Optional[int] = None

class AMIDataset(Dataset):
    def __init__(self, jsonl_file, feature_extractor, tokenizer):
        self.data = []
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

        with open(jsonl_file, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        audio_path = item['source']
        text = item['target']

        # Load and resample audio
        speech, sr = sf.read(audio_path)
        inputs = self.feature_extractor(speech, sampling_rate=sr, return_tensors="pt")
        
        # Tokenize text
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(text, return_tensors="pt").input_ids.squeeze(0)

        return {
            'input_values': inputs.input_values.squeeze(0),
            'attention_mask': inputs.attention_mask.squeeze(0),
            'labels': labels
        }

class WavLMTrainer:
    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DataTrainingArguments,
        training_args: TrainingArguments
    ):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args

        # Initialize tokenizer
        self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
            "facebook/wav2vec2-base",
            cache_dir=model_args.cache_dir,
            token="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"
        )

        self.config = WavLMConfig.from_pretrained(
            model_args.model_name_or_path,
            vocab_size=len(self.tokenizer),
            cache_dir=model_args.cache_dir,
            token="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"
        )

        self.model = WavLMForCTC.from_pretrained(
            model_args.model_name_or_path,
            config=self.config,
            cache_dir=model_args.cache_dir,
            token="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"
        ).cuda()

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            token="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"
        )

        self.train_dataset = AMIDataset(data_args.train_file, self.feature_extractor, self.tokenizer)
        self.eval_dataset = AMIDataset(data_args.validation_file, self.feature_extractor, self.tokenizer)
        self.test_dataset = AMIDataset(data_args.test_file, self.feature_extractor, self.tokenizer)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_args.learning_rate,
            weight_decay=training_args.weight_decay
        )

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.max_steps
        )

        wandb.init(project="wavlm-ami-asr", name=training_args.run_name)

    def train(self):
        best_eval_loss = float('inf')

        for epoch in range(self.training_args.num_train_epochs):
            self.model.train()
            train_loss = 0
            train_steps = 0

            train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.training_args.per_device_train_batch_size,
                shuffle=True
            )

            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")

            for batch in progress_bar:
                outputs = self.model(
                    input_values=batch['input_values'].to(self.model.device),
                    attention_mask=batch['attention_mask'].to(self.model.device),
                    labels=batch['labels'].to(self.model.device)
                )

                loss = outputs.loss
                train_loss += loss.item()
                train_steps += 1

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                progress_bar.set_postfix({'loss': loss.item()})

                wandb.log({
                    'train_loss': loss.item(),
                    'learning_rate': self.scheduler.get_last_lr()[0]
                })

            eval_loss = self.evaluate()

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                self.save_model('best_model')

            if epoch % self.training_args.save_steps == 0:
                self.save_model(f'checkpoint-{epoch}')

    def evaluate(self, test=False):
        self.model.eval()
        eval_loss = 0
        eval_steps = 0

        dataset = self.test_dataset if test else self.eval_dataset
        dataloader = DataLoader(
            dataset,
            batch_size=self.training_args.per_device_eval_batch_size
        )

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                outputs = self.model(
                    input_values=batch['input_values'].to(self.model.device),
                    attention_mask=batch['attention_mask'].to(self.model.device),
                    labels=batch['labels'].to(self.model.device)
                )

                eval_loss += outputs.loss.item()
                eval_steps += 1

        eval_loss = eval_loss / eval_steps

        wandb.log({
            'eval_loss' if not test else 'test_loss': eval_loss
        })

        return eval_loss

    def save_model(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Saved model checkpoint to {output_dir}")

def main():
    model_args = ModelArguments()
    data_args = DataTrainingArguments()
    training_args = TrainingArguments(
        output_dir="./wavlm-ami-asr-output",
        num_train_epochs=3,
        per_device_train_batch_size=4,  # Reduced batch size due to longer sequences
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        learning_rate=3e-5,
        run_name="wavlm-large-ami-asr"
    )

    trainer = WavLMTrainer(model_args, data_args, training_args)
    trainer.train()

    logger.info("Testing best model...")
    test_loss = trainer.evaluate(test=True)
    logger.info(f"Test loss: {test_loss}")

if __name__ == "__main__":
    main()
