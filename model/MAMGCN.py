# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils import scaled_Laplacian, cheb_polynomial
from lib.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp_multi
from lib.gat import GraphAttentionTransformerEncoder, MLP
from lib.FourierCorrelation import FourierBlock
from lib.AutoCorrelation import AutoCorrelationLayer
from lib.bi_mamba2 import BiMamba2_1D
from lib.Embed import BERTTimeEmbedding, PositionalEmbedding, FixedEmbedding, DoWEmbedding
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

        self.decomp = series_decomp_multi(kernel_size)

        # 季节
        self.te = BERTTimeEmbedding(max_position_embeddings=self.seq_len * 2, embedding_dim=self.enc_dim)
        self.gat = GraphAttentionTransformerEncoder(1, self.enc_dim, self.enc_dim, self.num_heads, dropout=self.dropout)
        self.fat = FourierBlock(in_channels=self.enc_dim,
                                        out_channels=self.enc_dim,
                                        seq_len=self.seq_len,
                                        modes=4,
                                        mode_select_method='random')
        attn_layers = [
            EncoderLayer(
                AutoCorrelationLayer(self.fat, self.enc_dim, self.num_heads),
                self.enc_dim,
                self.enc_dim * 4,
                moving_avg=kernel_size,
                dropout=self.dropout,
                activation='gelu')
        ]
        self.attn_layers = nn.ModuleList(attn_layers)

        self.tat = Encoder(self.attn_layers, norm_layer=my_Layernorm(self.enc_dim))
        # 趋势
        self.SAt = Spatial_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        self.cheb_conv_SAt = cheb_conv_withSAt(K, cheb_polynomials, in_channels, nb_chev_filter)
        self.mamba = BiMamba2_1D(self.enc_dim, self.enc_dim, self.enc_dim//2)

        # self.TAt = Temporal_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        #
        # self.time_conv = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 3), stride=(1, time_strides), padding=(0, 1))
        # self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        # self.ln = nn.LayerNorm(nb_time_filter)  #需要将channel放到最后一个维度上

    def forward(self, x, TE, SE, adj):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, nb_time_filter, T)
        '''
        adj = adj.to(x.device)
        batch_size, node, num_of_timesteps, num_of_features = x.shape

        #分解
        seasonal_init, trend_init = self.decomp(x.reshape(-1, num_of_timesteps, num_of_features))
        #趋势
        trend_init = trend_init.reshape(-1, node, num_of_timesteps, num_of_features) + TE + SE

        spatial_gate = self.SAt(trend_init.transpose(2, 3))
        trend_enc = self.cheb_conv_SAt(trend_init.transpose(2, 3), spatial_gate) # b n f l
        trend_enc = self.mamba(trend_enc.reshape(-1, num_of_features, num_of_timesteps))
        trend_enc = trend_enc.reshape(batch_size, node, num_of_features, num_of_timesteps).transpose(2, 3) # b n l f


        #季节
        seasonal_init = seasonal_init.reshape(-1, node, num_of_timesteps, num_of_features)
        rte = self.te(seasonal_init)
        seasonal_init = seasonal_init + rte + SE
        seasonal_enc = self.gat(seasonal_init.transpose(1, 2), adj).transpose(1, 2) # b n l f
        seasonal_enc, attn = self.tat(seasonal_enc.reshape(-1, num_of_timesteps, num_of_features))
        seasonal_enc = seasonal_enc.reshape(-1, node, num_of_timesteps, num_of_features)
        x_residual = x + seasonal_enc + trend_enc

        return x_residual


class MAMGCN_decblock(nn.Module):

    def __init__(self, DEVICE, in_channels, K, nb_chev_filter, num_heads, dropout, cheb_polynomials, num_of_vertices, num_of_timesteps):
        super(MAMGCN_decblock, self).__init__()
        self.enc_dim = in_channels
        self.num_heads = num_heads
        self.dropout = dropout
        self.seq_len = num_of_timesteps
        # 空间
        self.SAt = Spatial_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        self.cheb_conv_SAt = cheb_conv_withSAt(K, cheb_polynomials, in_channels, nb_chev_filter)
        # 时间
        self.TransformerDecoderLayer = nn.TransformerDecoderLayer(
            d_model=self.enc_dim,
            nhead=self.num_heads,
            dim_feedforward=self.enc_dim * 4, batch_first=True)
        self.nat = nn.TransformerDecoder(self.TransformerDecoderLayer, num_layers=1)

    def forward(self, x, x_enc):

        batch_size, node, total_of_timesteps, num_of_features = x.shape
        spatial_gate = self.SAt(x[..., :self.seq_len, :].transpose(2, 3))
        x_dec = self.cheb_conv_SAt(x.transpose(2, 3), spatial_gate).reshape(-1, total_of_timesteps, num_of_features)  # b n f l
        x_enc = x_enc.reshape(-1, self.seq_len, num_of_features)
        x_dec = self.nat(x_dec, x_enc).reshape(-1, node, total_of_timesteps, num_of_features)
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
        self.adj = torch.from_numpy(adj)
        self.emb = nn.Linear(1, in_channels)

        self.relu = nn.ReLU()
        self.pe = PositionalEmbedding(in_channels, max_len=num_of_vertices)
        self.tod = nn.Linear(1, in_channels)
        self.dow = DoWEmbedding(in_channels)
        self.seq_len = len_input
        self.pred_len = num_for_predict
        self.enc_BlockList = nn.ModuleList([MAMGCN_encblock(DEVICE, in_channels, K, nb_chev_filter, num_heads, dropout, cheb_polynomials, num_of_vertices, len_input, kernel_size) for _ in range(nb_block)])
        self.dec_BlockList = nn.ModuleList([MAMGCN_decblock(DEVICE, in_channels, K, nb_chev_filter, num_heads, dropout, cheb_polynomials, num_of_vertices, len_input)
                                            for _ in range(nb_block)])

        self.final_proj = MLP(
                 in_channels,
                 1,
                 hidden_dim=in_channels//2,
                 hidden_layers=3,
                 dropout=dropout,
                 activation='relu')

        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        global x_enc, x_dec
        x_e = self.relu(self.emb(x[..., :self.seq_len, :1]))
        tod = self.relu(self.tod(x[..., 1:2]))
        dow = self.dow(x[..., 2])
        te = tod + dow
        pe = self.pe(x[..., :1])
        for block in self.enc_BlockList:
            x_enc = block(x_e, te[..., :self.seq_len, :], pe[..., :self.seq_len, :], self.adj)

        x_d = F.pad(x_e, (0, 0, 0, self.pred_len)) + pe + te
        for block in self.dec_BlockList:
            x_dec = block(x_d, x_enc)

        output = self.final_proj(x_dec[..., -self.pred_len:, :]).squeeze()
        output =  output * (self.std) + self.mean #(b,N,T)
        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)

        return output


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