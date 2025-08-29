#!/usr/bin/env python3

import json
import numpy as np
import soundfile as sf
import os
from pathlib import Path

def create_synthetic_audio(output_path, duration=3.0, sample_rate=16000):
    """Create a synthetic audio file with some noise"""
    # Generate white noise
    samples = int(duration * sample_rate)
    audio = np.random.normal(0, 0.1, samples).astype(np.float32)
    
    # Add some sine waves to make it more audio-like
    t = np.linspace(0, duration, samples)
    audio += 0.1 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    audio += 0.05 * np.sin(2 * np.pi * 880 * t)  # 880 Hz tone
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write audio file
    sf.write(output_path, audio, sample_rate)
    print(f"Created synthetic audio: {output_path}")

def fix_dataset(jsonl_file):
    """Fix dataset by creating synthetic audio files and updating paths"""
    base_dir = Path('/projects/van-speech-nlp/zhang.jinda1/src/mllm_experiments/slam-llm/data/test_run/audio')
    base_dir.mkdir(parents=True, exist_ok=True)
    
    updated_lines = []
    audio_counter = 0
    
    with open(jsonl_file, 'r') as f:
        for line_num, line in enumerate(f):
            data = json.loads(line.strip())
            
            # Create new audio filename
            audio_filename = f"sample_{audio_counter:03d}.wav"
            audio_path = base_dir / audio_filename
            
            # Create synthetic audio file
            create_synthetic_audio(str(audio_path))
            
            # Update the source path
            data['source'] = str(audio_path)
            
            updated_lines.append(json.dumps(data))
            audio_counter += 1
    
    # Write updated dataset
    backup_file = f"{jsonl_file}.backup"
    os.rename(jsonl_file, backup_file)
    print(f"Backed up original to: {backup_file}")
    
    with open(jsonl_file, 'w') as f:
        for line in updated_lines:
            f.write(line + '\n')
    
    print(f"Updated dataset: {jsonl_file}")

if __name__ == "__main__":
    # Fix all dataset files
    dataset_files = [
        '/projects/van-speech-nlp/zhang.jinda1/src/mllm_experiments/slam-llm/data/test_run/train.jsonl',
        '/projects/van-speech-nlp/zhang.jinda1/src/mllm_experiments/slam-llm/data/test_run/validation.jsonl',
        '/projects/van-speech-nlp/zhang.jinda1/src/mllm_experiments/slam-llm/data/test_run/test.jsonl'
    ]
    
    for dataset_file in dataset_files:
        if os.path.exists(dataset_file):
            fix_dataset(dataset_file)
        else:
            print(f"Dataset file not found: {dataset_file}")
    
    print("Dataset fixing completed!")