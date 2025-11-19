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
    encoder_n_convolutions: int = 3
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
            gain=torch.nn.init.calculate_gain(w_init_gain)
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
            padding = (kernel_size - 1) // 2

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
                        in_channels=config.encoder_embed_dim if i != 0 else config.character_embed_dim,
                        out_channels=config.encoder_embed_dim,
                        kernel_size=config.encoder_kernel_size,
                        stride=1,
                        padding=None,
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
        batch_size, channels, seq_len = x.shape
        if input_lengths is None:
            input_lengths = torch.full((batch_size,), fill_value=seq_len, device=x.device)

        for block in self.convolutions:
            x = block(x)
        x = x.transpose(1,2)
        x = pack_padded_sequence(x,input_lengths.cpu(),batch_first=True)
        outputs, _ = self.lstm(x)
        outputs, _ = pad_packed_sequence(outputs,batch_first=True)
        return outputs  

class Prenet(nn.Module):
    def __init__(self,input_dim,prenet_dim,prenet_depth,dropout_p=0.5):
        super(Prenet,self).__init__()
        self.dropout_p = dropout_p
        dims = [input_dim] + [prenet_dim for _ in range(prenet_depth)] 

        self.layers = nn.ModuleList()

        for in_dim, out_dim in zip(dims[:-1],dims[1:]):
            self.layers.append(
                nn.Sequential(
                    LinearNorm(in_features=in_dim,
                    out_features=out_dim,
                    bias=False,
                    w_init_gain="relu"),
                nn.ReLU()
                )
            )

    def forward(self,x):
        for layer in self.layers:
            x = F.dropout(layer(x),p=self.dropout_p,training=self.training)
        return x


class LocationLayer(nn.Module):
    def __init__(self,attention_n_filters,attention_kernel_size,attention_dim):
        super(LocationLayer,self).__init__()

        self.conv = ConvNorm(
            in_channels=2,
            out_channels=attention_n_filters,
            kernel_size=attention_kernel_size,
            padding="same",
            bias=False
        )
        self.proj = LinearNorm(attention_n_filters,attention_dim, bias=False,w_init_gain="tanh")

    def forward(self,attention_weights):
        attention_weights= self.conv(attention_weights).transpose(1,2)
        attention_weights = self.proj(attention_weights)
        return attention_weights




class LocationSensitiveAttention(nn.Module):
    def __init__(self,attention_dim,decoder_hidden_size,encoder_hidden_size,attention_n_filters,attention_kernel_size):
        super(LocationSensitiveAttention,self).__init__()
        self.in_proj = LinearNorm(decoder_hidden_size,attention_dim,bias=True,w_init_gain="tanh")
        self.enc_proj = LinearNorm(encoder_hidden_size,bias=True,w_init_gain="tanh")
        self.what_have_i_said = LocationLayer(
            attention_n_filters,
            attention_kernel_size,
            attention_dim
        )
        self.energy_proj = LinearNorm(attention_dim,1,bias=False,w_init_gain="tanh")
        self.reset()
    
    def reset(self):
        self.enc_proj_cache= None

    def _calculate_alignment_energies(self,mel_input,encoder_output,cumulative_attention_weights,mask=None):
        mel_proj = self.in_proj(mel_input).unsqueeze(1)
        if self.enc_proj_cache is None:
            self.enc_proj_cache = self.enc_proj(encoder_output)
        cumulative_attention_weights = self.what_have_i_said(cumulative_attention_weights)
        energies = torch.tanh(mel_proj+ self.enc_proj_cache + cumulative_attention_weights)
        energies = self.energy_proj(energies).squeeze(-1)

        if mask is not None:
            energies = energies.masked_fill(mask.bool(),-float("inf"))
        return energies

    def forward(self,mel_input,encoder_output,cumulative_attention_weights,mask=None):
        energies = self._calculate_alignment_energies(mel_input,encoder_output,cumulative_attention_weights,mask)

        attention_weights = F.softmax(energies,dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1),encoder_output).squeeze(1)

        return attention_context, attention_weights
        
        
class PostNet(nn.Module):
    """
    To take final generated Mel from LSTM and postprocess to allow for any missing details to be added in
    """
    def __init__(self,num_mels,postnet_num_convs=5,postnet_n_filters=512,postnet_kernel_size=5,postnet_dropout_p=0.5):
        super(PostNet,self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(
            nn.Sequential(
                ConvNorm(
                    num_mels,
                    postnet_n_filters,
                    kernel_size=postnet_kernel_size,
                    padding="same",
                    w_init_gain="tanh"
                ),
                nn.BatchNorm1d(postnet_n_filters),
                nn.Tanh(),
                nn.Dropout(postnet_dropout_p)
            )
        )
        for _ in range(postnet_num_convs - 2):
            self.convs.append(
                nn.Sequential(
                    ConvNorm(postnet_n_filters,num_mels,kernel_size=postnet_kernel_size,padding="same"),
                    nn.BatchNorm1d(num_mels),
                    nn.Dropout(postnet_dropout_p)
                )

            )
    def forward(self,x):
        x = x.transpose(1,2)
        for conv_block in self.convs:
            x = conv_block(x)
        x = x.transpose(1,2)
        return x 

        










