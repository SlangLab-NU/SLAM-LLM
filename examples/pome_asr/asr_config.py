from dataclasses import dataclass, field
from typing import Optional, List
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
RUN_DIR = os.getenv('RUN_DIR')
if not RUN_DIR:
    raise ValueError("RUN_DIR environment variable not set in .env file")

# Get the current time
current_time = time.strftime("%Y-%m-%d_%H-%M-%S")

from torch.distributed.fsdp import ShardingStrategy

@dataclass
class ModelConfig:
    file: str = "examples/pome_asr/model/slam_model_asr.py:model_factory"
    llm_name: str = "vicuna-13b-v1.5"
    llm_path: str = "PATH/to/LLAMA/7B"
    llm_type: str = "decoder_only"
    llm_dim: int = 4096
    llm_inference_config: str = "repetition_penalty"

    encoder_name: Optional[str] = None
    encoder_ds_rate: int = 2
    encoder_path: Optional[str] = None
    encoder_dim: int = 1280
    
    encoder_projector: str = "linear"
    encoder_projector_ds_rate: int = 5  # downsample rate
    qformer_layers: int = 8

    modal: str = "audio"
    normalize: Optional[bool] = field(default=False, metadata={
        "help": "whether input is normalized, used for models such as wavlm"
    })
    encoder_type: str = field(default="finetune", metadata={
        "help": "whether model is only pretrained or finetuned, used for models such as hubert"
    })

    # j: update second encoder configs
    encoder2_name: Optional[str] = None
    encoder2_dim: Optional[int] = 1024
    encoder2_path: Optional[str] = None

    identifier: Optional[str] = None  # j: add identifier

@dataclass
class PeftConfig:
    peft_method: str = "lora" # None , llama_adapter, prefix
    r: int = 8
    lora_alpha: int = 32
    target_modules: List[str] = field(default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "gate_proj",
            "down_proj",
            ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    lora_dropout: float = 0.05
    inference_mode: bool = False

@dataclass
class TrainConfig:
    model_name:str = "PATH/to/LLAMA/7B"
    enable_ddp:bool = False
    enable_deepspeed:bool = False
    enable_fsdp:bool = False
    low_cpu_fsdp:bool = False
    run_validation:bool = True
    batch_size_training:int = 4
    batching_strategy:str = field(default="packing", metadata={
        "help":"alternative: padding"
    }) #
    context_length:int = 4096
    gradient_accumulation_steps:int = 1
    num_epochs:int = 3
    resume_step: int = 0 # j: resume step and epoch
    resume_epoch: int = 0

    num_workers_dataloader:int = 1
    warmup_steps:int = 1000
    total_steps:int = 100000
    validation_interval:int = 1000 # j: adjust to 3000
    lr:float = 1e-4
    weight_decay:float = 0.0
    gamma:float = 0.85
    seed:int = 42
    use_fp16:bool = False
    mixed_precision:bool = True
    val_batch_size:int = 1

    use_peft:bool = False
    peft_config:PeftConfig = field(default_factory=PeftConfig)
    output_dir:str = "PATH/to/save/PEFT/model"
    freeze_layers:bool = False
    num_freeze_layers:int = 1
    quantization:bool = False
    one_gpu:bool = False
    save_model:bool = True
    dist_checkpoint_root_folder:str = "PATH/to/save/FSDP/model" # will be used if using FSDP
    dist_checkpoint_folder:str = "fine-tuned" # will be used if using FSDP
    save_optimizer:bool = False # will be used if using FSDP
    use_fast_kernels:bool = False # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    run_test_during_validation:bool = False
    run_test_during_validation_file:str = "test.wav"
    run_test_during_validation_prompt:str = "<|ASR|>"
    freeze_llm:bool = field(default=False, metadata={
        "help": "whether to freeze llm when finetuning, should be true when use peft finetuning"
    })
    freeze_encoder:bool = False
    freeze_encoder2:bool = False # j: update freeze encoder2
    # New field: Save embedding
    save_embedding: bool = False # j: inference plotting for speech langanguage embeddings

    # Add flag_test functionality flag
    test_flag: bool = field(default=True, metadata={
        "help": "functionality flag for testing, default is True"
    })



@dataclass
class DataConfig:
    dataset: str = "speech_dataset"
    file: str = "src/slam_llm/datasets/speech_dataset.py:get_speech_dataset"
    train_data_path: Optional[str] = None
    val_data_path: Optional[str] = None
    train_split: str = "train"
    test_split:str = "validation"
    prompt: Optional[str] = None
    data_path: Optional[str] = None
    max_words: Optional[int] = None
    max_mel: Optional[float] = None
    fix_length_audio: int = -1
    inference_mode:bool = False
    input_type: str = field(default="raw", metadata={
                                "help":"Use raw when input is wav, mel when for whisper"
                            })
    mel_size: int = field(default=80, metadata={
        "help": "80 for whisper large v1 and v2, 128 for v3"
    })
    normalize: Optional[bool] = field(default=False, metadata={
        "help": "whether input is normalized, used for models such as wavlm"
    })

@dataclass
class FSDPConfig:
    mixed_precision: bool = True
    use_fp16: bool = False
    # sharding_strategy = "FULL_SHARD" #ShardingStrategy = ShardingStrategy.FULL_SHARD
    sharding_strategy: ShardingStrategy = "NO_SHARD" #ShardingStrategy.NO_SHARD #MZY: set NO_SHARD when use DDP
    checkpoint_type: str = "SHARDED_STATE_DICT"  # alternatively can use SHARDED_STATE_DICT save one file per rank, and can resize the world-size.
    fsdp_activation_checkpointing: bool = True
    fsdp_cpu_offload: bool = False
    pure_bf16: bool = False
    optimizer: str = "AdamW"

@dataclass
class LogConfig:
    use_wandb: bool = True
    wandb_dir: str = os.path.join(RUN_DIR, "out/log/wandb_log")
    wandb_entity_name: str = "jindaz-work"
    wandb_project_name: str = "SLAM-LLM"
    wandb_exp_name: str = "exp_name"
    log_file: str = os.path.join(RUN_DIR, f"out/log/{current_time}.txt")
    log_interval: int = 5
