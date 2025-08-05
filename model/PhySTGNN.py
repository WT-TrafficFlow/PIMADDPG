# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils import scaled_Laplacian, cheb_polynomial
from lib.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp_multi
from lib.gat import GraphAttentionTransformerEncoder, MLP, GateGCN
from lib.FourierCorrelation import FourierBlock
from lib.AutoCorrelation import AutoCorrelationLayer
from lib.bi_mamba2 import BiMamba2_1D
from lib.Embed import BERTTimeEmbedding, PositionalEmbedding, FixedEmbedding, DoWEmbedding, TokenEmbedding, SpatioTemporalEmbedding
from lib.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from lib.Transformer_EncDec import Decoder
from torch import Tensor
from typing import Optional
from lib.mamba import Mamba, MambaConfig



class BI_Mamba(nn.Module):
    def __init__(self, in_channels):
        super(BI_Mamba, self).__init__()
        self.mamba_Config = MambaConfig(d_model=in_channels, n_layers=1)
        self.mamba_for = Mamba(self.mamba_Config)
        self.mamba_back = Mamba(self.mamba_Config)
        self.proj = nn.Linear(in_channels*2, in_channels)
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(in_channels)  #需要将channel放到最后一个维度上

    def forward(self, x, bi_direction=True):
        if bi_direction:
            x1 = self.mamba_for(x)
            x2 = self.mamba_back(x.flip(1)).flip(1)
            y = torch.cat([x1, x2], dim=-1)
            y = self.relu(self.proj(y))
            # y = self.proj(x1, x2)
            x = self.ln(x+y)
        else:
            x = self.mamba2_for(x)
        return x


class TensorFusionFFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TensorFusionFFN, self).__init__()
        # 定义FFN（前馈网络）
        self.ffn = nn.Sequential(
            nn.Linear(input_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        # 定义批量归一化
        self.bn = nn.BatchNorm2d(output_dim)

    def forward(self, tensor1, tensor2):
        # 确保两个输入张量形状一致（B, N, T, F）
        assert tensor1.shape == tensor2.shape, "Input tensors must have the same shape"

        # 拼接两个张量，形状变为 (B, N, T, F*2)
        fused_tensor = torch.cat((tensor1, tensor2), dim=-1)  # 拼接沿着最后一维

        # 通过FFN处理
        ffn_output = self.ffn(fused_tensor)

        # 对FFN的输出进行BN处理
        bn_output = self.bn(ffn_output.permute(0, 3, 1, 2))  # 对 (B, F, N, T) 进行BN

        # 恢复为原形状 (B, N, T, F)
        bn_output = bn_output.permute(0, 2, 3, 1)  # 恢复为 (B, N, T, F)


        return bn_output

class TrafficFlowModule(nn.Module):
    def __init__(self, N, mode=1, embedding_dim=1):
        super(TrafficFlowModule, self).__init__()

        self.N = N  # Number of sites
        self.embedding_dim = embedding_dim
        self.mode = mode  # Select the mode (1, 2, or 3)

        # Mode 1 parameters: v_f and k_m for each station
        if self.mode == 1:
            self.v_f_1 = nn.Parameter(torch.randn(N, 1))  # v_f for Formula 1, [N, 1]
            self.k_m_1_inv = nn.Parameter(torch.randn(N, 1))  # k_m inverse for Formula 1, [N, 1]

        # Mode 2 parameters: v_0 and k_f for each station
        elif self.mode == 2:
            self.v_0_2 = nn.Parameter(torch.randn(N, 1))  # v_0 for Formula 2, [N, 1]
            self.k_f_2_inv = nn.Parameter(torch.randn(N, 1))  # k_f inverse for Formula 2, [N, 1]

        # Mode 3 parameters: v_f, k_c, m for each station
        elif self.mode == 3:
            self.v_f_3 = nn.Parameter(torch.randn(N, 1))  # v_f for Formula 3, [N, 1]
            self.k_c_3_inv = nn.Parameter(torch.randn(N, 1))  # k_c inverse for Formula 3, [N, 1]
            self.m_3 = nn.Parameter(torch.randn(N, 1))  # m for Formula 3, [N, 1]

        else:
            raise ValueError("Invalid mode selected. Please choose between 1, 2, or 3.")

    def forward(self, k):
        """
        k: 输入的密度张量，形状为 [B, N, T, 1]，B 是批量大小，N 是站点数量，T 是时间步数
        """
        B, N, T, _ = k.shape

        if self.mode == 1:
            # Formula 1: q = k * v_f * exp(-k / k_m)
            q = k * self.v_f_1.unsqueeze(0).unsqueeze(2) * torch.exp(-k * self.k_m_1_inv.unsqueeze(0).unsqueeze(2))

        elif self.mode == 2:
            # Formula 2: q = v_0 * k * ln(k_f / k)
            q = self.v_0_2.unsqueeze(0).unsqueeze(2) * k * torch.log(self.k_f_2_inv.unsqueeze(0).unsqueeze(2) / k)

        elif self.mode == 3:
            # Formula 3: q = k * v_f / [1 + (k / k_c)^m]^(2 / m)
            k_c_ratio = k * self.k_c_3_inv.unsqueeze(0).unsqueeze(2)
            q = k * self.v_f_3.unsqueeze(0).unsqueeze(2) / (
                        1 + torch.pow(k_c_ratio, self.m_3.unsqueeze(0).unsqueeze(2))) ** (
                            2 / self.m_3.unsqueeze(0).unsqueeze(2))

        return q

class FusionGate(nn.Module):
    def __init__(self, input_dim):
        super(FusionGate, self).__init__()
        # 用于生成gate的线性层
        self.gate = nn.Linear(input_dim * 2, input_dim)  # 假设两个输入维度相同，乘以2表示拼接后
        self.sigmoid = nn.Sigmoid()  # 用于将gate值限制在(0, 1)范围内
        self.relu = nn.ReLU()

    def forward(self, input1, input2):
        # 假设input1和input2的维度相同：[batch_size, input_dim]

        # 拼接两个输入 [batch_size, input_dim * 2]
        combined_input = torch.cat([input1, input2], dim=-1)

        # 通过线性层生成gate值，大小为[batch_size, 1]
        gate_value = self.relu(self.gate(combined_input))

        # 对两个输入进行加权 [batch_size, input_dim]
        # output = gate_value * input1 + (1 - gate_value) * input2
        output = gate_value
        return output


class CustomTransformerDecoderLayerNoSelfAttention(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", batch_first: bool = False):
        super(CustomTransformerDecoderLayerNoSelfAttention, self).__init__(d_model, nhead, dim_feedforward, dropout,
                                                                           activation, batch_first=batch_first)

    def forward(
            self,
            tgt: Tensor,
            memory: Tensor,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            tgt_is_causal: bool = False,
            memory_is_causal: bool = False,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer, excluding self-attention.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
            memory_is_causal: If specified, applies a causal mask as ``memory mask``.

        Shape:
            see the docs in Transformer class.
        """

        x = tgt
        if self.norm_first:
            # Removed self-attention (_sa_block)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            x = x + self._ff_block(self.norm3(x))
        else:
            # Removed self-attention (_sa_block)
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal))
            x = self.norm3(x + self._ff_block(x))

        return x


class Spatial_Attention_layer(nn.Module):
    '''
    compute spatial attention scores
    '''
    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Spatial_Attention_layer, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(num_of_timesteps).to(DEVICE))
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_timesteps).to(DEVICE))
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.bs = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices).to(DEVICE))
        self.Vs = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices).to(DEVICE))


    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B,N,N)
        '''

        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)  # (b,N,F,T)(T)->(b,N,F)(F,T)->(b,N,T)

        rhs = torch.matmul(self.W3, x).transpose(-1, -2)  # (F)(b,N,F,T)->(b,N,T)->(b,T,N)

        product = torch.matmul(lhs, rhs)  # (b,N,T)(b,T,N) -> (B, N, N)

        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))  # (N,N)(B, N, N)->(B,N,N)

        S_normalized = F.softmax(S, dim=1)

        return S_normalized


class cheb_conv_withSAt(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv_withSAt, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])

    def forward(self, x, spatial_attention):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):

                T_k = self.cheb_polynomials[k]  # (N,N)

                T_k_with_at = T_k.mul(spatial_attention)   # (N,N)*(N,N) = (N,N) 多行和为1, 按着列进行归一化

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = T_k_with_at.permute(0, 2, 1).matmul(graph_signal)  # (N, N)(b, N, F_in) = (b, N, F_in) 因为是左乘，所以多行和为1变为多列和为1，即一行之和为1，进行左乘

                output = output + rhs.matmul(theta_k)  # (b, N, F_in)(F_in, F_out) = (b, N, F_out)

            outputs.append(output.unsqueeze(-1))  # (b, N, F_out, 1)

        return F.relu(torch.cat(outputs, dim=-1))  # (b, N, F_out, T)


class Temporal_Attention_layer(nn.Module):
    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Temporal_Attention_layer, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(num_of_vertices).to(DEVICE))
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices).to(DEVICE))
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.be = nn.Parameter(torch.FloatTensor(1, num_of_timesteps, num_of_timesteps).to(DEVICE))
        self.Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps).to(DEVICE))

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B, T, T)
        '''
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)
        # x:(B, N, F_in, T) -> (B, T, F_in, N)
        # (B, T, F_in, N)(N) -> (B,T,F_in)
        # (B,T,F_in)(F_in,N)->(B,T,N)

        rhs = torch.matmul(self.U3, x)  # (F)(B,N,F,T)->(B, N, T)

        product = torch.matmul(lhs, rhs)  # (B,T,N)(B,N,T)->(B,T,T)

        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B, T, T)

        E_normalized = F.softmax(E, dim=1)

        return E_normalized


class cheb_conv(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])

    def forward(self, x):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):

                T_k = self.cheb_polynomials[k]  # (N,N)

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = graph_signal.permute(0, 2, 1).matmul(T_k).permute(0, 2, 1)

                output = output + rhs.matmul(theta_k)

            outputs.append(output.unsqueeze(-1))

        return F.relu(torch.cat(outputs, dim=-1))


class MAMGCN_encblock(nn.Module):

    def __init__(self, DEVICE, in_channels, K, nb_chev_filter, num_heads, dropout, cheb_polynomials, num_of_vertices, num_of_timesteps, kernel_size):
        super(MAMGCN_encblock, self).__init__()
        self.enc_dim = in_channels
        self.num_heads = num_heads
        self.dropout = dropout
        self.seq_len = num_of_timesteps
        # 分解

        # self.decomp = series_decomp_multi(kernel_size)

        # 季节
        # self.te = BERTTimeEmbedding(max_position_embeddings=self.seq_len * 2, embedding_dim=self.enc_dim)
        self.gat = GraphAttentionTransformerEncoder(1, self.enc_dim, self.enc_dim, self.num_heads, dropout=self.dropout)
        # self.fat = FourierBlock(in_channels=self.enc_dim,
        #                                 out_channels=self.enc_dim,
        #                                 seq_len=self.seq_len,
        #                                 modes=4,
        #                                 mode_select_method='random')
        # self.fat = MultiWaveletTransform(ich=self.enc_dim, L=3, base='legendre')
        # attn_layers = [
        #     EncoderLayer(
        #         AutoCorrelationLayer(self.fat, self.enc_dim, self.num_heads),
        #         self.enc_dim,
        #         self.enc_dim * 4,
        #         moving_avg=kernel_size,
        #         dropout=self.dropout,
        #         activation='gelu')
        # ]
        # self.attn_layers = nn.ModuleList(attn_layers)
        #
        # self.tat = Encoder(self.attn_layers, norm_layer=my_Layernorm(self.enc_dim))

        self.TransformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.enc_dim,
            nhead=self.num_heads,
            dim_feedforward=self.enc_dim * 4, batch_first=True)
        self.tat = nn.TransformerEncoder(self.TransformerEncoderLayer, num_layers=1)
        self.gate = FusionGate(self.enc_dim)

        self.sffn = TensorFusionFFN(self.enc_dim, self.enc_dim*2, self.enc_dim)
        self.tffn = TensorFusionFFN(self.enc_dim, self.enc_dim * 2, self.enc_dim)




        # 趋势
        # self.TAt = Temporal_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        # self.SAt = Spatial_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        # self.cheb_conv_SAt = cheb_conv_withSAt(K, cheb_polynomials, in_channels, nb_chev_filter)
        self.GateGCN = GateGCN(in_channels, in_channels, in_channels, 2)
        # self.mamba = BiMamba2_1D(self.enc_dim, self.enc_dim, self.enc_dim)
        self.mamba = BI_Mamba(self.enc_dim)
        # self.t_gate = FusionGate(self.enc_dim)
        # self.TAt = Temporal_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        #
        # self.time_conv = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 3), stride=(1, time_strides), padding=(0, 1))
        # self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self.ln = nn.LayerNorm(self.enc_dim)  #需要将channel放到最后一个维度上

    def forward(self, x, STE, adj):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, nb_time_filter, T)
        '''
        batch_size, node, num_of_timesteps, num_of_features = x.shape
        # STE = TE + SE
        x_mamba_in = x
        x_enc = x + STE
        x_mamba_out = self.mamba(x_mamba_in.reshape(-1, num_of_timesteps, num_of_features), bi_direction=True).reshape(-1, node, num_of_timesteps, num_of_features)
        x_tat_out = self.tat(x_enc.reshape(-1, num_of_timesteps, num_of_features)).reshape(-1, node, num_of_timesteps, num_of_features)
        x_gcn_in = x_enc.transpose(1, 2).reshape(-1, node, num_of_features)

        x_gcn_out = self.GateGCN(x_gcn_in, adj).reshape(-1, num_of_timesteps, node,
                                                              num_of_features).transpose(1, 2)
        x_sat_in = x_enc.transpose(1, 2)
        x_sat_out = self.gat(x_sat_in, adj).transpose(1, 2)  # b n l f
        x_s = self.sffn(x_gcn_out, x_sat_out)
        x_t = self.tffn(x_mamba_out, x_tat_out)
        x_ = self.gate(x_s, x_t)

        x_residual = x + x_
        return x_residual



class MAMGCN_decblock(nn.Module):

    def __init__(self, DEVICE, in_channels, K, nb_chev_filter, num_heads, dropout, cheb_polynomials, num_of_vertices, num_of_timesteps):
        super(MAMGCN_decblock, self).__init__()
        self.enc_dim = in_channels
        self.num_heads = num_heads
        self.dropout = dropout
        self.seq_len = num_of_timesteps
        # 空间
        self.GateGCN = GateGCN(in_channels, in_channels, in_channels, 2)
        # self.SAt = Spatial_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        # self.cheb_conv_SAt = cheb_conv_withSAt(K, cheb_polynomials, in_channels, nb_chev_filter)
        # 空间交叉
        # self.CrossLayer = CustomTransformerDecoderLayerNoSelfAttention(
        #     d_model=self.enc_dim,
        #     nhead=self.num_heads,
        #     dim_feedforward=self.enc_dim * 4, batch_first=True)
        # self.cat = nn.TransformerDecoder(self.CrossLayer, num_layers=1)


        # 时间
        self.TransformerDecoderLayer = nn.TransformerDecoderLayer(
            d_model=self.enc_dim,
            nhead=self.num_heads,
            dim_feedforward=self.enc_dim * 4, batch_first=True)
        self.nat = nn.TransformerDecoder(self.TransformerDecoderLayer, num_layers=1)
        self.ln = nn.LayerNorm(self.enc_dim)  # 需要将channel放到最 后一个维度上

    def forward(self, x, x_enc, adj):

        batch_size, node, total_of_timesteps, num_of_features = x.shape
        # spatial_gate = self.SAt(x[..., :self.seq_len, :].transpose(2, 3))
        # x_dec = self.cheb_conv_SAt(x.transpose(2, 3), spatial_gate).transpose(2, 3)
        x_dec = x.transpose(1, 2).reshape(-1, node, num_of_features)
        x_dec = self.GateGCN(x_dec, adj).reshape(-1, total_of_timesteps, node,
                                                                          num_of_features).transpose(1, 2)  # b n l f
        # x_dec_in = x_dec[..., :self.seq_len, :].transpose(1, 2).reshape(-1, node, num_of_features)
        # x_enc_s = x_enc.transpose(1, 2).reshape(-1, node, num_of_features)
        # x_dec_s = self.cat(x_dec_in, x_enc_s).reshape(-1, self.seq_len, node, num_of_features).transpose(1, 2) #b n 12 f
        # x_dec = torch.cat([x_dec_s, x_dec[..., self.seq_len:, :]], dim=-2) #b n 24 f
        x_dec = x_dec.reshape(-1, total_of_timesteps, num_of_features)
        x_enc = x_enc.reshape(-1, self.seq_len, num_of_features)
        x_dec = self.nat(x_dec, x_enc).reshape(-1, node, total_of_timesteps, num_of_features)
        # x_enc = x_enc.reshape(-1, node, self.seq_len, num_of_features).transpose(1, 2).reshape(-1, node, num_of_features)
        # x_dec = x_dec.transpose(1, 2).reshape(-1, node, num_of_features)
        # x_dec = self.csat(x_dec, x_enc).reshape(-1, total_of_timesteps, node, num_of_features).transpose(1, 2)
        x_dec = x + x_dec

        return x_dec





class MAMGCN_submodule(nn.Module):

    def __init__(self, DEVICE, nb_block, in_channels, K, nb_chev_filter, num_heads, dropout, cheb_polynomials, num_for_predict, len_input, num_of_vertices, mean, std,
                 kernel_size=None, adj=None):


        '''
        :param nb_block:
        :param in_channels:
        :param K:
        :param nb_chev_filter:
        :param nb_time_filter:
        :param time_strides:
        :param cheb_polynomials:
        :param nb_predict_step:
        '''

        super(MAMGCN_submodule, self).__init__()
        if kernel_size is None:
            kernel_size = [2, 3]
        self.mean = mean
        self.std = std
        self.adj = torch.from_numpy(adj).float().to(DEVICE)
        # self.emb = nn.Linear(1, in_channels)
        self.emb = MLP(
            5, in_channels,
            hidden_dim=in_channels // 2,
            hidden_layers=2,
            dropout=dropout,
            activation='relu')
        self.enc_dim = in_channels
        self.relu = nn.ReLU()
        self.node = num_of_vertices
        self.ve = TokenEmbedding(in_channels, in_channels)
        self.pe = BERTTimeEmbedding(max_position_embeddings=num_of_vertices, embedding_dim=in_channels)
        # self.tod = MLP(
        #     1, in_channels,
        #     hidden_dim=in_channels // 2,
        #     hidden_layers=2,
        #     dropout=dropout,
        #     activation='relu')
        # self.dow = DoWEmbedding(in_channels)
        self.ste = SpatioTemporalEmbedding(max_num_vehicles=12, max_time_steps=13, embedding_dim=in_channels)
        self.seq_len = len_input
        self.pred_len = num_for_predict
        self.enc_BlockList = nn.ModuleList([MAMGCN_encblock(DEVICE, in_channels, K, nb_chev_filter, num_heads, dropout, cheb_polynomials, num_of_vertices, len_input, kernel_size) for _ in range(nb_block)])
        self.dec_BlockList = nn.ModuleList([MAMGCN_decblock(DEVICE, in_channels, K, nb_chev_filter, num_heads, dropout, cheb_polynomials, num_of_vertices, len_input)
                                            for _ in range(nb_block)])
        self.enc_ctrl = MLP(
            1, in_channels,
            hidden_dim=in_channels // 2,
            hidden_layers=2,
            dropout=dropout,
            activation='relu')
        self.final_proj = MLP(
                 in_channels,
                 2,
                 hidden_dim=in_channels//2,
                 hidden_layers=2,
                 dropout=dropout,
                 activation='relu')
        # self.final_proj = nn.Linear(in_channels, 1)
        self.FD_module = TrafficFlowModule(N=num_of_vertices, mode=1, embedding_dim=1)
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        global x_enc, x_dec
        x_flow = x
        # x_ = x[..., :self.seq_len, :1]
        # mean_enc = x_.mean(2, keepdim=True).detach()   # b n 1 1
        # x_flow = x_flow - mean_enc
        # std_enc = torch.sqrt(torch.var(x_, dim=2, keepdim=True, unbiased=False) + 1e-5).detach()
        # x_flow = x_flow / std_enc
        # x_emb = self.relu(self.emb(x_flow))
        x_emb = self.emb(x_flow)
        x_e = x_emb[..., :self.seq_len, :]
        # x_e = x_e.reshape(-1, self.seq_len, self.enc_dim)
        # x_e = self.ve(x_e).reshape(-1, self.node, self.seq_len, self.enc_dim)
        # tod = self.tod(x[..., 1:2])
        # dow = self.dow(x[..., 2])
        # te = tod + dow
        # pe = self.relu(self.pe(x_emb))
        STE = self.ste(x_flow)
        # x_e = x_e + te[..., :self.seq_len, :] + pe[..., :self.seq_len, :]
        for block in self.enc_BlockList:
            x_e = block(x_e, STE[..., :self.seq_len, :], self.adj)
        x_ctrl_q = self.enc_ctrl(x_flow[..., self.seq_len:, -1:])
        x_d = torch.cat([x_emb[..., :self.seq_len, :], x_ctrl_q], dim=2) + STE
        # x_d = F.pad(x_emb[..., :self.seq_len, :], (0, 0, 0, self.pred_len)) + STE
        for block in self.dec_BlockList:
            x_d = block(x_d, x_e, self.adj)

        output = self.relu(self.final_proj(x_d[..., -self.pred_len:, :]))
        fd_q = self.FD_module(output[..., -1:])
        # output = output * std_enc + mean_enc
        # output = output.squeeze()
        # output = output * (self.std) + self.mean #(b,N,T)
        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)

        return output, fd_q


def make_model(DEVICE, nb_block, in_channels, K, nb_chev_filter, num_heads, dropout, adj_mx, num_for_predict, len_input, num_of_vertices, mean, std, kernel_size):
    '''

    :param DEVICE:
    :param nb_block:
    :param in_channels:
    :param K:
    :param nb_chev_filter:
    :param nb_time_filter:
    :param time_strides:
    :param cheb_polynomials:
    :param nb_predict_step:
    :param len_input
    :return:
    '''
    L_tilde = scaled_Laplacian(adj_mx)
    cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(DEVICE) for i in cheb_polynomial(L_tilde, K)]
    model = MAMGCN_submodule(DEVICE, nb_block, in_channels, K, nb_chev_filter, num_heads, dropout, cheb_polynomials, num_for_predict,
         len_input, num_of_vertices, mean, std, kernel_size, adj_mx)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    return model