import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

seed = 42
torch.random.manual_seed(seed)
np.random.seed(seed)


class Encoder(nn.Module):
    """Base class for encoders"""
    
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, src, src_mask=None):
        """
        Args:
            src      : source of shape `(batch, src_len)`
            src_mask : mask indicating the lengths of each source of shape `(batch, time)`
        """
        raise NotImplementedError


class AudioEncoder(Encoder):
    """Base class for audio encoders"""
    
    def __init__(self, **kwargs):
        super().__init__()
        self.cnn = []
        self.rnn = []
        self.strides = [1]
        self.stride = 1
        last_input_features = kwargs['input_dim']
        if kwargs['audio_input'] == 'raw':
            for l in kwargs['conv']:
                o, k, s, p = l
                self.cnn.append(nn.Conv1d(last_input_features, o, k, stride=s, padding=p)) # Torch does not support 'same' padding for stride > 1
                self.cnn.append(nn.BatchNorm1d(o))
                self.cnn.append(nn.ReLU())
                self.stride *= s
                self.strides.append(self.stride)
                last_input_features = o

            for l in kwargs['gru']:
                unit = l
                self.rnn.append(nn.GRU(last_input_features, unit[0], batch_first=True))
                last_input_features = unit[0]

        self.cnn = nn.ModuleList(self.cnn)
        self.rnn = nn.ModuleList(self.rnn)

        self.dense = nn.Linear(last_input_features, kwargs['fc'])
        self.act = nn.LeakyReLU()

    
    def forward(self, src, src_mask=None, verbose=False):
        """
        Args:
            src         : source of shape `(batch, time, feature)`
            src_mask    : mask indicating the lengths of each source of shape `(batch, time)`
        """
        # keep the batch mask
        if src_mask is not None:
            mask = src_mask[:,::self.stride]
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
        else:
            mask = None
        
        # [B, T, F]
        # cnn
        x = src.transpose(1, 2)
        for i, layer in enumerate(self.cnn): # [B, F, T] -> [B, Conv1d, T/self.stride]
            x = layer(x)
        x = x.transpose(1, 2)
        
        # rnn
        hidden = None
        for layer in self.rnn: # [B, T/self.stride, Conv1D] -> [B, T/self.stride, GRU]
            if mask is not None:
                x = x * mask
            x, hidden = layer(x, hidden)
        
        x = self.dense(x)      # [B, T/self.stride, Dense]
        
        LD = x
        x = self.act(x)
        
        x = torch.nan_to_num(x) * mask
        mask = mask.squeeze(-1)

        return x, LD, mask


class FusionFiLM(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.scale_gen = nn.Linear(in_dim, in_dim)
        self.shift_gen = nn.Linear(in_dim, in_dim)

    def forward(self, x, cond):
        # x: (B, T, D), cond: (B, T, D)
        scale = self.scale_gen(cond)
        shift = self.shift_gen(cond)
        return x * scale + shift


class EfficientAudioEncoder(Encoder):
    """Efficient encoder class for audio encoders"""
    
    def __init__(self, downsample=True, **kwargs):
        super().__init__()
        self.downsample = downsample
        self.layer = [] 
        self.deConv = None
        last_input_features = kwargs['input_dim']
        self.film_fusioning = kwargs['film_fusion']
        self.vad = kwargs['vad']
        self.vad_layer = []

        if self.downsample:
            self.layer.append(nn.Conv1d(last_input_features, kwargs['fc'], 5, stride=2, padding=2))
            self.layer.append(nn.BatchNorm1d(kwargs['fc']))
            self.layer.append(nn.ReLU())
            self.layer.append(nn.MaxPool1d(2))
            self.layer.append(nn.Conv1d(kwargs['fc'], kwargs['fc'], 5, stride=2, padding=2))
            self.layer.append(nn.BatchNorm1d(kwargs['fc']))
            self.layer.append(nn.ReLU())
            self.dense = nn.Linear(96, kwargs['fc'])
        else:
            self.layer.append(nn.Conv1d(last_input_features, kwargs['fc'], 3, stride=2, padding=1))
            self.layer.append(nn.BatchNorm1d(kwargs['fc']))
            self.layer.append(nn.ReLU())
            self.layer.append(nn.Conv1d(kwargs['fc'], kwargs['fc'], 3, stride=1, padding=1))
            self.layer.append(nn.BatchNorm1d(kwargs['fc']))
            self.layer.append(nn.ReLU())
            self.deConv = nn.ConvTranspose1d(96, kwargs['fc'], 5, 4) # 96: MAGIC NUM defined by google speech embedding
        self.layer = nn.Sequential(*self.layer)
        
        self.act = nn.LeakyReLU()

        if self.film_fusioning:
            self.film_fusion = FusionFiLM(kwargs['fc'])
        
        if self.vad:
            self.vad_layer.append(nn.AdaptiveAvgPool1d(1))
            self.vad_layer.append(nn.Linear(kwargs['fc'], 1))
            self.vad_layer.append(nn.Sigmoid())

    def forward(self, src, src_mask=None, verbose=False):
        """
        Args:
            src         : (spectrogram, gembed) where
                        : spectrogram - log mel-spectrogram of shape `(batch, time, mel)`
                        : gembed      - google speech embedding of shape `(batch, time / 8, 96)`
            src_mask    : mask indicating the lengths of spectrogram of shape `(batch, time)`
        """        
        spectrogram, gembed = src

        if src_mask is not None:
            s_mask, g_mask = src_mask
            if self.downsample:
                mask = s_mask[:,::8]
            else:
                mask = s_mask[:,::2]
            gembed = torch.nan_to_num(gembed) * g_mask.unsqueeze(-1)
        else:
            mask = None

        x = spectrogram.transpose(1, 2)
        x = self.layer(x)
        x = x.transpose(1, 2)

        LD = x

        # [B, T/8, dense] or [B, T/2, dense]
        if self.film_fusioning:
            if self.downsample:
                y = self.act(self.dense(gembed))  # (B, T/8, D)
                if x.shape[1] > y.shape[1]:
                    y = F.pad(y, (0, 0, 0, x.shape[1] - y.shape[1]), value=0.0)
                elif x.shape[1] < y.shape[1]:
                    y = y[:, :x.shape[1], :]
            else:
                y = gembed.transpose(1, 2)  # (B, 96, T/8)
                y = self.act(self.deConv(y))  # (B, D, T)
                y = y.transpose(1, 2)
                if x.shape[1] > y.shape[1]:
                    y = F.pad(y, (0, 0, 0, x.shape[1] - y.shape[1]), value=0.0)
                elif x.shape[1] < y.shape[1]:
                    y = y[:, :x.shape[1], :]

            x = self.film_fusion(x, y)

        else:
            if self.downsample:
                y = self.act(self.dense(gembed))

                # Summation two embedding
                x = x + nn.functional.pad(y, (0, 0, 0, x.shape[1] - y.shape[1], 0, 0), value=0.0)
            else:
                y = gembed.transpose(1, 2)
                y = self.act(self.deConv(y))
                y = y.transpose(1, 2)

                if x.shape[1] > y.shape[1]:
                    x = x + nn.functional.pad(y, (0, 0, 0, x.shape[1] - y.shape[1], 0, 0), value=0.0)
                elif x.shape[1] < y.shape[1]:
                    x = x + y[:, :x.shape[1], :]
                else:
                    x = x + y
        
        x = torch.nan_to_num(x) * mask.unsqueeze(-1)  # (B, time, embedding)
        LD = torch.nan_to_num(LD) * mask.unsqueeze(-1)

        if self.vad:
            z = x.permute(0, 2, 1)  # (B, embedding, time)
            z = self.vad_layer(z)  # (B, 1)

        return x, LD, mask, z


class TextEncoder(Encoder):
    """Base class for text encoders"""
    
    def __init__(self, **kwargs):
        super().__init__()
        
        self.features = kwargs['text_input']
        self.vocab = kwargs['vocab']
        if self.features == 'phoneme':
            self.dense = nn.Linear(kwargs['vocab'], kwargs['fc'])
        elif self.features == 'g2p_embed':
            self.dense = nn.Linear(256, kwargs['fc'])
        self.act = nn.LeakyReLU()

    def forward(self, src, verbose=False):
        """
        Args:
            src         : phoneme token of shape `(batch, phoneme, *)`
                        : [WARNING] for 'g2p_embed' features, shape is `(batch, phoneme, 256)`
            src_mask    : mask indicating the lengths of each source of shape `(batch, time)`
        """
        # [B, phoneme] -> [B, phoneme, embedding]
        x = src
        src_mask = (src != 0.0)
        if src_mask.dim() == 3:
            src_mask = src_mask[:,:,0]
        
        if self.features == 'phoneme':
            x = nn.functional.one_hot(x.to(torch.int64), num_classes=self.vocab).to(torch.float)
        x = self.act(self.dense(x))
        x = torch.nan_to_num(x) * src_mask.unsqueeze(-1)

        return x, src_mask