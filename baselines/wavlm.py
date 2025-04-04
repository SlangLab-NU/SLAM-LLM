import torch
from transformers import WavLMForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer
from datasets import load_dataset, DatasetDict, Audio
import pandas as pd
import librosa
import jiwer
from transformers import TrainingArguments, Trainer, DataCollatorCTCWithPadding

def load_custom_dataset(audio_dir, transcript_file, split='train'):
    """
    加载自定义数据集，从音频目录和转录CSV文件。
    CSV应有'file'（相对audio_dir的路径）和'text'列。
    """
    df = pd.read_csv(transcript_file)
    dataset = DatasetDict({
        split: load_dataset('csv', data_files={'train': transcript_file}, split='train')
    })
    dataset[split] = dataset[split].cast_column("audio", Audio(sampling_rate=16000))
    return dataset

def preprocess_function(examples, processor):
    """
    预处理音频文件和转录文本为WavLM。
    重采样到16kHz并编码标签。
    """
    audio = [librosa.load(f, sr=16000)[0] for f in examples['file']]
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with processor.as_target_processor():
        labels = processor(text=examples['text'], return_tensors="pt", padding=True).input_ids
    return {"input_values": inputs.input_values, "labels": labels}

def main():
    # 可配置参数
    model_name = "microsoft/wavlm-large"  # 可更改为"microsoft/wavlm-large"以获得更好性能
    task = "asr"  # 当前设为ASR，可扩展为其他任务
    audio_dir = "/path/to/audio"  # 更新为您的音频目录
    transcript_file = "/path/to/transcripts.csv"  # 更新为您的CSV文件路径

    # 加载数据集
    dataset = load_custom_dataset(audio_dir, transcript_file)
    
    # 初始化处理器和分词器
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_name)
    
    # 预处理数据集
    dataset = dataset.map(lambda x: preprocess_function(x, processor), batched=True, remove_columns=dataset.column_names['train'])
    
    # 加载模型
    model = WavLMForCTC.from_pretrained(model_name, ctc_loss_reduction="mean")
    model.freeze_feature_extractor()  # 冻结特征提取器以提高效率
    
    # 定义数据整理器
    data_collator = DataCollatorCTCWithPadding(processor=processor)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir="./wavlm-finetuned",
        per_device_train_batch_size=32,
        num_train_epochs=30,
        learning_rate=1e-4,
        weight_decay=0.005,
        warmup_steps=1000,
        save_steps=500,
        eval_steps=500,
        fp16=True,  # 使用混合精度以加快训练
        gradient_checkpointing=True,
        group_by_length=True,  # 按长度分批以提高效率
        push_to_hub=True,  # 保存到Hugging Face Hub
    )
    
    # 初始化训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['train'],  # 简单起见；如果有测试集，请添加
        data_collator=data_collator,
        compute_metrics=lambda pred: {"wer": jiwer.wer(pred.label_ids, pred.predictions)}
    )
    
    # 训练模型
    trainer.train()
    
    # 评估并打印结果
    results = trainer.evaluate()
    print(f"最终WER: {results['eval_wer']}")

if __name__ == "__main__":
    main()