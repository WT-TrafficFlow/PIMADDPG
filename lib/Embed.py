import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x_pe = self.pe[:, :x.size(1)]  # 取出 self.pe 的前 x.size(1) 个元素
        x_pe = x_pe.unsqueeze(2) # 增加两个维度以匹配 x 的维度数量
        x_pe = x_pe.repeat(x.size(0), 1, x.size(2), 1)  # 在每个维度上重复
        return x_pe









class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class DoWEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='hh'):
        super(DoWEmbedding, self).__init__()
        weekday_size = 7
        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        self.weekday_embed = Embed(weekday_size, d_model)


    def forward(self, x):
        x = x.long()
        weekday_x = self.weekday_embed(x)
        return weekday_x

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)

class DataEmbedding_onlypos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_onlypos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
    
class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # try:
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        # except:
        #     a = 1
        return self.dropout(x)

class BERTTimeEmbedding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim):
        super(BERTTimeEmbedding, self).__init__()
        self.embeddings = nn.Embedding(max_position_embeddings, embedding_dim)

    def forward(self, input_ids):
        position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).unsqueeze(2).expand(*(input_ids.shape[0], 1, input_ids.shape[2]))
        return self.embeddings(position_ids)


class SpatioTemporalEmbedding(nn.Module):
    def __init__(self, max_num_vehicles=12, max_time_steps=13, embedding_dim=64):
        super(SpatioTemporalEmbedding, self).__init__()

        self.max_num_vehicles = max_num_vehicles
        self.max_time_steps = max_time_steps
        self.embedding_dim = embedding_dim

        # 车辆编号的空间位置编码
        self.vehicle_position_embedding = nn.Embedding(max_num_vehicles, embedding_dim)

        # 时间位置编码
        self.time_position_embedding = nn.Embedding(max_time_steps, embedding_dim)

        # 通过BERT的方式进行位置嵌入
        self.position_embedding = nn.Embedding(max_num_vehicles * max_time_steps, embedding_dim)

    def forward(self, x):
        """
        x: 输入的张量，形状为 [B, N, T, F]，其中B是批次大小，N是车辆数量，T是时间步长，F是特征维度。
        """
        B, N, T, F = x.shape

        # 生成车辆编号的空间位置编码 [B, N]
        vehicle_ids = torch.arange(N, device=x.device).unsqueeze(0).repeat(B, 1)  # [B, N]
        vehicle_embeddings = self.vehicle_position_embedding(vehicle_ids)  # [B, N, embedding_dim]

        # 生成时间步长的时间位置编码 [B, T]
        time_steps = torch.arange(T, device=x.device).unsqueeze(0).repeat(B, 1)  # [B, T]
        time_embeddings = self.time_position_embedding(time_steps)  # [B, T, embedding_dim]

        # 扩展并整合空间和时间位置编码 [B, N, T, embedding_dim]
        spatio_temporal_embeddings = vehicle_embeddings.unsqueeze(2) + time_embeddings.unsqueeze(1)


        return spatio_temporal_embeddings