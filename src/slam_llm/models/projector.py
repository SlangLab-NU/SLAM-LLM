import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

#j: add dual encoder projector
class EncoderProjectorDualConcat(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = config.encoder_projector_ds_rate
        self.encoder_dim1 = config.encoder_dim  # Dimension of first encoder
        self.encoder_dim2 = config.encoder2_dim  # Dimension of second encoder
        self.llm_dim = config.llm_dim

        # Calculate the combined dimension after concatenation
        combined_dim = (self.encoder_dim1 + self.encoder_dim2) * self.k

        # Define the layers for projection
        self.linear1 = nn.Linear(combined_dim, 2048)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2048, self.llm_dim)

    def forward(self, x, gradient_checkpoint=False):
        batch_size, seq_len, dim = x.size()
        
        # Calculate the new sequence length after removing excess frames
        num_frames_to_discard = x.size(1) % self.k
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]

        # Update seq_len after discarding frames
        seq_len = x.size(1)

        # Downsample the sequence
        x = x.contiguous()
        x = x.view(batch_size, seq_len // self.k, dim * self.k)  # Concatenating the features for the two encoders

        # Apply the first linear layer
        if gradient_checkpoint:
            x = checkpoint(self.linear1, x)  # Use gradient checkpointing for linear1
        else:
            x = self.linear1(x)

        # Apply ReLU activation
        x = self.relu(x)

        # Apply the second linear layer
        if gradient_checkpoint:
            x = checkpoint(self.linear2, x)  # Use gradient checkpointing for linear2
        else:
            x = self.linear2(x)

        return x


class EncoderProjectorConcat(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = config.encoder_projector_ds_rate
        self.encoder_dim = config.encoder_dim
        self.llm_dim = config.llm_dim
        self.linear1 = nn.Linear(self.encoder_dim * self.k, 2048)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2048, config.llm_dim)
        self.saved_speech_embeddings = None # j: save embeddings 

    def forward(self, x, gradient_checkpoint=False):
        batch_size, seq_len, dim = x.size()
        num_frames_to_discard = seq_len % self.k
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)
        
        x = x.contiguous()
        x = x.view(batch_size, seq_len // self.k, dim * self.k)
        self.saved_speech_embeddings = x.detach().cpu()
        
        if gradient_checkpoint:
            x = checkpoint(self.linear1, x)
        else:
            x = self.linear1(x)
        x = self.relu(x)
        if gradient_checkpoint:
            x = checkpoint(self.linear2, x)
        else:
            x = self.linear2(x)
        return x


class EncoderProjectorCov1d(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = config.encoder_projector_ds_rate
        self.encoder_dim = config.encoder_dim
        self.llm_dim = config.llm_dim
        self.conv1d = nn.Conv1d(in_channels=self.encoder_dim, out_channels=self.encoder_dim, kernel_size=self.k, stride=self.k, padding=0)
        self.linear1 = nn.Linear(self.encoder_dim, 2048)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(2048, self.llm_dim)
        self.relu2 = nn.ReLU()
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = x.transpose(1, 2)
        x = self.relu1(x)
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear2(x)
        return x

class EncoderProjectorQFormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder_dim = config.encoder_dim
        self.llm_dim = config.llm_dim
        from transformers import Blip2QFormerConfig, Blip2QFormerModel
        configuration = Blip2QFormerConfig()
        configuration.encoder_hidden_size = self.encoder_dim
        configuration.num_hidden_layers = config.qformer_layers

        self.query_len = int(config.get("query_len", 64))
        self.query = nn.Parameter(torch.zeros(1, self.query_len, configuration.hidden_size))
        self.query.data.normal_(mean=0.0, std=1.0)
        self.qformer = Blip2QFormerModel(configuration)

        self.linear = nn.Linear(configuration.hidden_size, self.llm_dim)
        self.norm = nn.LayerNorm(self.llm_dim, eps=1e-5)

    def forward(self, x, atts):
        query = self.query.expand(x.shape[0], -1, -1)
        
        query_output = self.qformer(
            query_embeds=query,
            encoder_hidden_states=x,
            encoder_attention_mask=atts,
            return_dict=True,
        )
        
        query_proj = self.norm(self.linear(query_output.last_hidden_state))
        
        return query_proj