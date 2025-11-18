import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from dataclasses import dataclass

@dataclass
class Tacotron2Config:

    ## Mel Input Features
    num_mels: int = 80
    ## Character Embedding
    character_embed_dim: int = 512
    num_chars: int = 67
    pad_token_id: int = 0

    ## Encoder config
    encoder_kernel_size:int = 5
    encoder_n_convolutons: int = 3
    encoder_embed_dim: int = 512
    encoder_dropout_p: float = 0.5

    ## Decoder Config
    decoder_embed_dim: int = 1024
    decoder_prenet_dim: int = 256
    decoder_prenet_depth: int = 2
    decoder_postnet_num_convs: int = 5
    decoder_postnet_kernel_size: int = 5
    decoder_postnet_dropout_p: float = 0.5
    decoder_dropout_p:float = 0.1

    ## Attention Config
    attention_dim: int = 128
    attention_location_n_filters: int = 32
    attention_location_kernel_size: int = 31
    attention_dropout_p: float = 0.1

class LinearNorm(nn.Module):
    def __init__(self,
    in_features,
    out_features,
    bias=True,
    w_init_gain="linear"):
        super(LinearNorm,self).__init__()
        self.linear = nn.Linear(in_features,out_features,bias=bias)
        torch.nn.init.xavier_uniform_(
            self.linear.weight,
            gain=torch.nn.init.calcuate_gain(w_init_gain)
        )

    def forward(self,x):
        return self.linear(x)


class ConvNorm(nn.Module):
    """
    Standard Convolution layer with different initialization strategies

    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear"

    ):
        super(ConvNorm,self).__init__()
        if padding is None:
            pading = "same"

        self.conv = nn.Conv1d(
            in_channels,out_channels,kernel_size=kernel_size,
            stride=stride,padding=padding, dilation=dilation,
            bias=bias
        )
        torch.nn.init.xavier_uniform_(
            self.conv.weight,gain=torch.nn.init.calculate_gain(w_init_gain)
        )
    def forward(self,x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self,config):
        super(Encoder,self).__init__()
        self.config = config
        self.embeddings = nn.Embedding(config.num_chars,config.character_embed_dim,padding_idx=config.pad_token_id)
        self.convolutions = nn.ModuleList()

        for i in range(config.encoder_n_convolutions): 
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(
                        in_channels=config.encoder_embed_dim if  i != 0 else config.character_embed_dim,
                        out_channels=config.encoder_embed_dim,
                        kernel_size=config.encoder_kernel_size,
                        stride=1,
                        padding="same",
                        dilation=1,
                        w_init_gain="relu"

                    ),
                    nn.BatchNorm1d(config.encoder_embed_dim),
                    nn.ReLU(),
                    nn.Dropout(config.encoder_dropout_p)
                )
            )
        self.lstm = nn.LSTM(
            input_size=config.encoder_embed_dim,
            hidden_size=config.encoder_embed_dim//2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
    
    def forward(self,x,input_lengths=None):
        x = self.embeddings(x).transpose(1,2)
        batch_size, channels,seq_len = x.shape
        if input_lengths is None:
            torch.full((batch_size,),fill_value=seq_len,device=x.device)

        for block in self.blocks:
            x = block(x)
        x = x.transpose(1,2)
        x = pack_padded_sequence(x,input_lengths.cpu(),batch_first=True)
        outputs, _ = self.lstm(x)
        outputs, _ = pad_packed_sequence(outputs,batch_first=True)
        return outputs


