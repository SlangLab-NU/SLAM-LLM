import os
import types
import torch
import soundfile as sf
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import List, Optional, Tuple, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from slam_llm.utils.config_utils import generate_peft_config
from slam_llm.utils.train_utils import print_module_size, print_model_size
from peft import PeftModel, PeftConfig
from torch.nn import CrossEntropyLoss
from slam_llm.utils.metric import compute_accuracy

import logging
logger = logging.getLogger(__name__)


def model_factory(train_config, model_config, **kwargs):
    # return necessary components for training
    tokenizer = setup_tokenizer(train_config, model_config, **kwargs)

    encoder = setup_encoder(train_config, model_config, **kwargs)

    # llm
    llm = setup_llm(train_config, model_config, **kwargs)

    # projector
    encoder_projector = setup_encoder_projector(
        train_config, model_config, **kwargs
    )
    model = slam_model(
        encoder,
        llm,
        encoder_projector,
        tokenizer,
        train_config,
        model_config,
        **kwargs,
    )

    # FIX(MZY): load model ckpt(mainly projector, related to model_checkpointing/checkpoint_handler.py: save_model_checkpoint_peft)
    ckpt_path = kwargs.get("ckpt_path", None)
    if ckpt_path is not None:
        logger.info("loading other parts from: {}".format(ckpt_path))
        try:
            ckpt_dict = torch.load(ckpt_path, map_location="cpu")
            
            # Check for mismatch in checkpoint and model state_dict keys
            model_state_dict = model.state_dict()
            missing_keys, unexpected_keys = model_state_dict.keys() - ckpt_dict.keys(), ckpt_dict.keys() - model_state_dict.keys()
            if missing_keys or unexpected_keys:
                logger.error(f"Checkpoint and model state_dict mismatch:")
                logger.error(f"Missing keys: {missing_keys}")
                logger.error(f"Unexpected keys: {unexpected_keys}")
                raise ValueError(f"Checkpoint does not match model architecture. Missing or unexpected keys.")

            # Load the state dict into the model
            model.load_state_dict(ckpt_dict, strict=False)

            logger.info("Model loaded successfully from checkpoint.")

        except Exception as e:
            logger.error(f"Error loading checkpoint from {ckpt_path}: {e}")
            raise RuntimeError(f"Failed to load checkpoint: {e}")

    print_model_size(model, train_config, int(
        os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)
    return model, tokenizer


def setup_tokenizer(train_config, model_config, **kwargs):
    # Load the tokenizer and add special tokens
    if "vallex" in model_config.llm_name.lower():
        return None
    elif "mupt" in model_config.llm_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_config.llm_path,
                                                  trust_remote_code=True,
                                                  use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.llm_path, trust_remote_code=True)
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def get_encoder_instance(encoder_name, model_config):
    if encoder_name == "whisper" or encoder_name == "qwen-audio":
        from slam_llm.models.encoder import WhisperWrappedEncoder
        return WhisperWrappedEncoder.load(model_config)
    elif encoder_name == "beats":
        from slam_llm.models.encoder import BEATsEncoder
        return BEATsEncoder.load(model_config)
    elif encoder_name == "eat":
        from slam_llm.models.encoder import EATEncoder
        return EATEncoder.load(model_config)
    elif encoder_name == "SpatialAST":
        from slam_llm.models.encoder import SpatialASTEncoder
        return SpatialASTEncoder.load(model_config)
    elif encoder_name == "wavlm":
        from slam_llm.models.encoder import WavLMEncoder
        return WavLMEncoder.load(model_config)
    elif encoder_name == "av_hubert":
        from slam_llm.models.encoder import AVHubertEncoder
        return AVHubertEncoder.load(model_config)
    elif encoder_name == "hubert":
        from slam_llm.models.encoder import HubertEncoder
        return HubertEncoder.load(model_config)
    elif encoder_name == "musicfm":
        from slam_llm.models.encoder import MusicFMEncoder
        return MusicFMEncoder.load(model_config)
    elif encoder_name == "emotion2vec":
        from slam_llm.models.encoder import Emotion2vecEncoder
        encoder = Emotion2vecEncoder.load(model_config)
    elif "w2v" or "w2p" in encoder_name.lower():
        from slam_llm.models.encoder import Wav2Vec2Encoder
        return Wav2Vec2Encoder.load(model_config)
    elif "llama" in encoder_name.lower():
        from slam_llm.models.encoder import HfTextEncoder
        return HfTextEncoder.load(model_config)
    else:
        return None


def setup_encoder(train_config, model_config, **kwargs):
    encoder_list = model_config.encoder_name.split(
        ",") if model_config.encoder_name else []
    if len(encoder_list) == 0:
        return None
    if len(encoder_list) == 1:
        encoder_name = encoder_list[0]
        encoder = get_encoder_instance(encoder_name, model_config)

    print_module_size(encoder, encoder_name, int(
        os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)

    if train_config.freeze_encoder:
        for name, param in encoder.named_parameters():
            param.requires_grad = False
        encoder.eval()
    print_module_size(encoder, encoder_name, int(
        os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)

    return encoder


def setup_encoder2(train_config, model_config, **kwargs):
    encoder_list = model_config.encoder2_name.split(
        ",") if model_config.encoder2_name else []

    encoder2_name = encoder_list[0]
    encoder2 = get_encoder_instance(encoder2_name, model_config)

    print_module_size(encoder2, encoder2_name, int(
        os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)

    if train_config.freeze_encoder2:
        for name, param in encoder2.named_parameters():
            param.requires_grad = False
        encoder2.eval()
    print_module_size(encoder2, encoder2_name, int(
        os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)

    return encoder2


def setup_llm(train_config, model_config, **kwargs):
    from pkg_resources import packaging
    use_cache = False if train_config.enable_fsdp or train_config.enable_ddp else None
    if (train_config.enable_fsdp or train_config.enable_ddp) and train_config.low_cpu_fsdp:
        """
        for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
        this avoids cpu oom when loading large models like llama 70B, in which case
        model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
        overhead and currently requires latest nightly.
        """
        # v = packaging.version.parse(torch.__version__)
        # verify_latest_nightly = v.is_devrelease and v.dev >= 20230701
        # if not verify_latest_nightly:
        #     raise Exception("latest pytorch nightly build is required to run with low_cpu_fsdp config, "
        #                     "please install latest nightly.")
        rank = int(os.environ["RANK"])
        if rank == 0:
            if "vallex" in model_config.llm_name.lower():
                from src.slam_llm.models.vallex.vallex_config import VallexConfig
                from src.slam_llm.models.vallex.vallex_model import VALLE
                vallex_config = VallexConfig(
                    **model_config
                )
                model = VALLE(vallex_config)
            elif "aya" in model_config.llm_name.lower():
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_config.llm_path,
                    load_in_8bit=True if train_config.quantization else None,
                    device_map="auto" if train_config.quantization else None,
                    use_cache=use_cache,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_config.llm_path,
                    load_in_8bit=True if train_config.quantization else None,
                    device_map="auto" if train_config.quantization else None,
                    use_cache=use_cache,
                )
        else:
            llama_config = AutoConfig.from_pretrained(model_config.llm_path)
            llama_config.use_cache = use_cache
            # with torch.device("meta"):
            if "aya" in model_config.llm_name.lower():
                model = AutoModelForSeq2SeqLM(llama_config)
            else:
                # (FIX:MZY): torch 2.0.1 does not support `meta`
                model = AutoModelForCausalLM(llama_config)

    else:
        if "vallex" in model_config.llm_name.lower():
            from src.slam_llm.models.vallex.vallex_config import VallexConfig
            from src.slam_llm.models.vallex.vallex_model import VALLE
            vallex_config = VallexConfig(
                **model_config
            )
            model = VALLE(vallex_config)
        elif "aya" in model_config.llm_name.lower():
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_config.llm_path,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
                use_cache=use_cache,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_config.llm_path,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
                use_cache=use_cache,
                trust_remote_code=True
            )
    if (train_config.enable_fsdp or train_config.enable_ddp) and train_config.use_fast_kernels:
        """
        For FSDP and FSDP+PEFT, setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels
        based on the hardware being used. This would speed up fine-tuning.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)
        except ImportError:
            logger.warning(
                "Module 'optimum' not found. Please install 'optimum' it before proceeding.")

    print_module_size(model, model_config.llm_name, int(
        os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)

    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_kbit_training(model)

    if train_config.freeze_llm:  # TODO:to test offical `freeze_layers` and `num_freeze_layers`
        for name, param in model.named_parameters():
            param.requires_grad = False
        model.eval()

    if kwargs.get("peft_ckpt", None):  # (FIX:MZY):reload will get wrong results when decoding
        logger.info("loading peft_ckpt from: {}".format(
            kwargs.get("peft_ckpt")))
        model = PeftModel.from_pretrained(
            model=model, model_id=kwargs.get("peft_ckpt"), is_trainable=True)
        model.print_trainable_parameters()
    elif train_config.use_peft:
        logger.info("setup peft...")
        peft_config = generate_peft_config(train_config)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    print_module_size(model, model_config.llm_name, int(
        os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)
    return model


def setup_encoder_projector(train_config, model_config, **kwargs):
    if model_config.encoder_projector == "linear":
        from slam_llm.models.projector import EncoderProjectorConcat
        encoder_projector = EncoderProjectorConcat(model_config)
    elif model_config.encoder_projector == "cov1d-linear":
        from slam_llm.models.projector import EncoderProjectorCov1d
        encoder_projector = EncoderProjectorCov1d(model_config)
    elif model_config.encoder_projector == "q-former":
        from slam_llm.models.projector import EncoderProjectorQFormer
        encoder_projector = EncoderProjectorQFormer(model_config)
    # j: add dual projectordual
    elif model_config.encoder_projector == "dual":
        from slam_llm.models.projector import EncoderProjectorDualConcat
        encoder_projector = EncoderProjectorDualConcat(model_config)
    else:
        return None
    print_module_size(encoder_projector, model_config.encoder_projector, int(
        os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)
    return encoder_projector


class slam_model(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        llm: nn.Module,
        encoder_projector: nn.Module,
        tokenizer,
        train_config,
        model_config,
        encoder2=None,  # j: New parameter for the second encoder
        **kwargs
    ):
        super().__init__()
        # modality encoder
        self.encoder = encoder
        # j: Optional second encoder
        self.encoder2 = encoder2

        # llm
        self.llm = llm

        # projector
        self.encoder_projector = encoder_projector

        # tokenizer
        self.tokenizer = tokenizer
        self.metric = kwargs.get("metric", "acc")

        self.train_config = train_config
        self.model_config = model_config

        if train_config.get("enable_deepspeed", False):
            def new_forward(self, input):
                output = F.layer_norm(
                    input.float(),
                    self.normalized_shape,
                    self.weight.float() if self.weight is not None else None,
                    self.bias.float() if self.bias is not None else None,
                    self.eps,
                )
                return output.type_as(input)
            for item in self.modules():
                if isinstance(item, nn.LayerNorm):
                    item.forward = types.MethodType(new_forward, item)

    def save_embeddings(self, speech_embeddings, language_embeddings):
        # Generate a unique filename (you might want to use a more sophisticated naming scheme)
        filename = f'/work/van-speech-nlp/jindaznb/jslpnb/mllm_experiments/slam-llm/examples/asr_librispeech/plot/embeddings/{self.model_config.identifier}.pt'

        # Save both embeddings in a single file
        torch.save({
            'speech_embeddings': speech_embeddings,
            'language_embeddings': language_embeddings
        }, filename)

        print(f"Embeddings saved to {filename}")


    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **kwargs,
                ):
        audio_mel = kwargs.get("audio_mel", None)
        audio_mel_mask = kwargs.get("audio_mel_mask", None)
        audio_mel_post_mask = kwargs.get(
            "audio_mel_post_mask", None)  # 2x downsample for whisper

        audio = kwargs.get("audio", None)
        audio_mask = kwargs.get("audio_mask", None)
        visual = kwargs.get("visual", None)
        visual_mask = kwargs.get("visual_mask", None)
        text = kwargs.get("text", None)

        # for text encoder
        instruct_ids = kwargs.get("instruct_ids", None)
        instruct_mask = kwargs.get("instruct_mask", None)

        modality_mask = kwargs.get("modality_mask", None)

        zh_data = kwargs.get("zh", None)
        en_data = kwargs.get("en", None)

        encoder_outs = None
        if audio_mel is not None or audio is not None or visual is not None or text is not None:
            if self.train_config.freeze_encoder:  # freeze encoder
                self.encoder.eval()
            if self.encoder2 and self.train_config.freeze_encoder2:  # freeze encoder2
                self.encoder2.eval()

            if self.model_config.encoder_name == "whisper":
                encoder_outs = self.encoder.extract_variable_length_features(
                    audio_mel.permute(0, 2, 1))  # bs*seq*dim
            if self.model_config.encoder_name == "beats":
                encoder_outs, audio_mel_post_mask = self.encoder.extract_features(
                    audio_mel, audio_mel_mask)  # bs*seq*dim
            if self.model_config.encoder_name == "eat":
                encoder_outs = self.encoder.model.extract_features(audio_mel.unsqueeze(
                    dim=1), padding_mask=None, mask=False, remove_extra_tokens=False)['x']
            if self.model_config.encoder_name == "clap":
                if text is not None:
                    encoder_outs = self.encoder.encode_text(
                        text).unsqueeze(1)  # [btz, 1, dim]
                elif audio is not None:
                    encoder_outs = self.encoder.encode_audio(
                        audio)  # with projection-based decoding
            if self.model_config.encoder_name == "SpatialAST":
                # output: [bs, seq_len=3+512, dim=768]
                encoder_outs = self.encoder(audio)
            if self.model_config.encoder_name == "wavlm":
                # (FIX:MZY): 1-audio_mask is needed for wavlm as the padding mask
                encoder_outs = self.encoder.extract_features(audio, 1 - audio_mask)
            if self.model_config.encoder_name == "hubert":
                results = self.encoder(source=audio, padding_mask=1-audio_mask)
                if self.model_config.encoder_type == "pretrain":
                    encoder_outs, audio_mel_post_mask = results["x"], results["padding_mask"]
                if self.model_config.encoder_type == "finetune":
                    encoder_outs, audio_mel_post_mask = results["encoder_out"], results["padding_mask"]
                    encoder_outs = encoder_outs.transpose(0, 1)
            if self.model_config.encoder_name == "av_hubert":
                results = self.encoder(
                    source={'video': visual, 'audio': audio}, padding_mask=visual_mask)  # bs*seq*dim
                encoder_outs, audio_mel_post_mask = results["encoder_out"], results["padding_mask"]
                encoder_outs = encoder_outs.transpose(0, 1)
                audio_mel_post_mask = (~audio_mel_post_mask).float()
            if self.model_config.encoder_name == 'musicfm':
                encoder_outs = self.encoder.extract_features(
                    audio, padding_mask=None)  # MusicFM doesn't support padding mask
            if self.encoder is None:
                encoder_outs = audio_mel if audio_mel is not None else audio
            if self.model_config.encoder_name == 'w2v2':
                encoder_outs = self.encoder.extract_features(
                    source=audio, attention_mask=attention_mask)
            if self.model_config.encoder_name == "emotion2vec":
                encoder_outs = self.encoder.extract_features(audio, None)[
                    'x']  # bs*seq*dim
    
            # j: concat embeddings
            if self.encoder2 is not None:
                if self.model_config.encoder2_name == 'w2v2':
                    logger.info("Getting encoder output from second encoder (w2v2)")
                    encoder2_outs = self.encoder2.extract_features(
                        source=audio, attention_mask=attention_mask)

                assert not torch.equal(encoder_outs, encoder2_outs), "Warning: encoder_outs and encoder2_outs are identical!"
                combined_encoder_outs = torch.cat((encoder_outs, encoder2_outs), dim=-1)
                encoder_outs = combined_encoder_outs  # Assign after verification

            # j: projector
            if self.model_config.encoder_projector == "q-former":
                encoder_outs = self.encoder_projector(
                    encoder_outs, audio_mel_post_mask)
            if self.model_config.encoder_projector == "linear":
                encoder_outs = self.encoder_projector(encoder_outs)
            if self.model_config.encoder_projector == "cov1d-linear":
                encoder_outs = self.encoder_projector(encoder_outs)
            # j: add dual encoder_out
            if self.model_config.encoder_projector == "dual":
                encoder_outs = self.encoder_projector(encoder_outs)


            # j: save embedding after the projector
            if self.train_config.save_embedding:
                speech_embeddings = self.encoder_projector.saved_speech_embeddings
                # Save language embeddings after projector
                language_embeddings = encoder_outs.detach().cpu()
                # Save embeddings to file
                self.save_embeddings(speech_embeddings, language_embeddings)


        if instruct_ids is not None:
            if self.encoder is not None:
                encoder_outs = self.encoder(
                    input_ids=instruct_ids, attention_mask=instruct_mask).last_hidden_state

            if self.model_config.encoder_projector == "q-former":
                encoder_outs = self.encoder_projector(
                    encoder_outs, instruct_mask)
            if self.model_config.encoder_projector == "linear":
                encoder_outs = self.encoder_projector(encoder_outs)
        
        if input_ids is not None:
            input_ids[input_ids == -1] = 0
            if isinstance(self.llm, T5ForConditionalGeneration):
                inputs_embeds = self.llm.shared(input_ids)
            else:
                if hasattr(self.llm.model, "embed_tokens"):
                    inputs_embeds = self.llm.model.embed_tokens(input_ids)
                elif hasattr(self.llm.model.model, "embed_tokens"):
                    inputs_embeds = self.llm.model.model.embed_tokens(
                        input_ids)
                else:
                    inputs_embeds = self.llm.model.model.model.embed_tokens(
                        input_ids)

        if modality_mask is not None:
            logger.info("modality encoder")
            modality_mask_start_indices = (
                modality_mask == True).float().argmax(dim=1)
            modality_lengths = torch.clamp(modality_mask.sum(
                dim=1), max=encoder_outs.shape[1]).tolist()

            encoder_outs_pad = torch.zeros_like(inputs_embeds)
            for i in range(encoder_outs.shape[0]):
                encoder_outs_pad[
                    i, modality_mask_start_indices[i]:modality_mask_start_indices[i]+modality_lengths[i]
                ] = encoder_outs[i][:modality_lengths[i]]

            inputs_embeds = encoder_outs_pad + \
                inputs_embeds * (~modality_mask[:, :, None])

        if kwargs.get("inference_mode", False):
            return inputs_embeds, attention_mask

        if zh_data is not None and en_data is not None:
            model_outputs, acc = self.llm(zh=zh_data, en=en_data)
        else:
            model_outputs = self.llm(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
            acc = -1
            if self.metric:
                with torch.no_grad():
                    preds = torch.argmax(model_outputs.logits, -1)
                    acc = compute_accuracy(
                        preds.detach()[:, :-1], labels.detach()[:, 1:], ignore_label=-100)

        return model_outputs, acc

    @torch.no_grad()
    def generate(self,
                 llm_config: dict,
                 input_ids: torch.LongTensor = None,
                 attention_mask: Optional[torch.Tensor] = None,
                 position_ids: Optional[torch.LongTensor] = None,
                 past_key_values: Optional[List[torch.FloatTensor]] = None,
                 inputs_embeds: Optional[torch.FloatTensor] = None,
                 labels: Optional[torch.LongTensor] = None,
                 use_cache: Optional[bool] = None,
                 output_attentions: Optional[bool] = None,
                 output_hidden_states: Optional[bool] = None,
                 return_dict: Optional[bool] = None,
                 **kwargs,
                 ):
        kwargs["inference_mode"] = True

        inputs_embeds, attention_mask = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        model_outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            # max_length=kwargs.get("max_length", 200),
            max_new_tokens=llm_config["max_new_tokens"],
            num_beams=llm_config["num_beams"],
            do_sample=llm_config["do_sample"],
            min_length=llm_config["min_length"],
            top_p=llm_config["top_p"],
            repetition_penalty=llm_config["repetition_penalty"],
            length_penalty=llm_config["length_penalty"],
            temperature=llm_config["temperature"],
            attention_mask=attention_mask,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )

        return model_outputs
