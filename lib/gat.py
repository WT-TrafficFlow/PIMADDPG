import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.utils import softmax

class MultiHeadGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads, dropout=0.2):
        super(MultiHeadGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features // num_heads  # 每个头的输出维度
        self.num_heads = num_heads

        # 为每个头创建线性变换和注意力计算
        self.W = nn.ModuleList([nn.Linear(in_features, self.out_features) for _ in range(num_heads)])
        self.a = nn.ModuleList([nn.Linear(2 * self.out_features, 1) for _ in range(num_heads)])

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU()
        self.elu = nn.ELU()

    def forward(self, x, adj):
        head_outputs = []
        for head in range(self.num_heads):
            h = self.W[head](x)  # 对每个头应用线性变换

            # 计算注意力系数
            a_input = torch.cat([h.unsqueeze(2).expand(-1, -1, adj.size(0), -1, -1),
                                 h.unsqueeze(3).expand(-1, -1, -1, adj.size(1), -1)], dim=-1)
            e = self.leakyrelu(self.a[head](a_input)).squeeze(-1)  # (batch_size, num_nodes, num_nodes)

            # 掩码处理，防止在无连接处的注意力
            mask = adj == 0
            e.masked_fill_(mask, float('-inf'))

            # 计算注意力权重
            attention = F.softmax(e, dim=-1)
            attention = self.dropout(attention)

            # 应用注意力权重到节点特征
            h_prime = torch.matmul(attention, h)
            head_outputs.append(h_prime)

        # 将所有头的输出拼接
        h_out = torch.cat(head_outputs, dim=-1)  # (batch_size, num_nodes, num_heads*out_features)
        self.elu(h_out)
        return h_out

class GraphAttentionTransformerEncoderLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads=8, dropout=0.1):
        super(GraphAttentionTransformerEncoderLayer, self).__init__()
        self.attention = MultiHeadGraphAttentionLayer(in_features, out_features, num_heads, dropout)
        self.ffn = nn.Linear(out_features, out_features)
        self.layer_norm1 = nn.LayerNorm(out_features)
        self.layer_norm2 = nn.LayerNorm(out_features)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, adj):
        # Multi-head graph attention
        att_out = self.attention(x, adj)
        att_out = self.dropout1(att_out)
        x = self.layer_norm1(x + att_out)

        # Feedforward network
        ffn_out = self.ffn(x)
        ffn_out = self.dropout2(ffn_out)
        x = self.layer_norm2(x + ffn_out)

        return x

class GraphAttentionTransformerEncoder(nn.Module):
    def __init__(self, num_layers, in_features, out_features, num_heads=4, dropout=0.1):
        super(GraphAttentionTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([GraphAttentionTransformerEncoderLayer(in_features, out_features, num_heads, dropout) for _ in range(num_layers)])

    def forward(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)
        return x




class BERTSpaceTimeEmbedding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, num_nodes):
        super(BERTSpaceTimeEmbedding, self).__init__()
        self.time_embeddings = nn.Embedding(max_position_embeddings, embedding_dim)
        self.space_embeddings = nn.Embedding(num_nodes, embedding_dim)

    def forward(self, input_ids):
        batch_size, sequence_length, num_nodes = input_ids.size(0), input_ids.size(-1), input_ids.size(-2)

        # Time embeddings
        position_ids = torch.arange(sequence_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, sequence_length)
        time_embeddings = self.time_embeddings(position_ids)

        # Space embeddings
        node_ids = torch.arange(num_nodes, dtype=torch.long, device=input_ids.device)
        node_ids = node_ids.unsqueeze(0).unsqueeze(1).expand(batch_size, sequence_length, num_nodes)
        space_embeddings = self.space_embeddings(node_ids)

        # Combine time and space embeddings
        space_time_embeddings = time_embeddings.unsqueeze(2) + space_embeddings

        return space_time_embeddings.transpose(1, 3)


class MLP(nn.Module):
    '''
    Multilayer perceptron to encode/decode high dimension representation of sequential data
    '''

    def __init__(self,
                 f_in,
                 f_out,
                 hidden_dim=128,
                 hidden_layers=3,
                 dropout=0.05,
                 activation='tanh'):
        super(MLP, self).__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise NotImplementedError

        layers = [nn.Linear(self.f_in, self.hidden_dim),
                  self.activation, nn.Dropout(self.dropout)]
        for i in range(self.hidden_layers - 2):
            layers += [nn.Linear(self.hidden_dim, self.hidden_dim),
                       self.activation, nn.Dropout(dropout)]

        layers += [nn.Linear(hidden_dim, f_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # x:     B x S x f_in
        # y:     B x S x f_out
        y = self.layers(x)
        return y


class GatedGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super(GatedGraphConv, self).__init__()

        # Learnable weights for gating mechanism
        self.W = nn.Linear(in_channels, out_channels)
        self.U = nn.Linear(out_channels, out_channels)
        self.V = nn.Linear(out_channels, out_channels)

        # Attention mechanism
        self.attention = nn.Linear(2 * out_channels, 1)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(out_channels)

    def forward(self, x, adj):
        """
        x: Tensor of shape (batch_size, num_nodes, in_channels)
        adj: Tensor of shape (num_nodes, num_nodes)  -> 2D adjacency matrix shared across batches
        """
        batch_size, num_nodes, _ = x.shape

        # Linear transformation on input features
        h = self.W(x)  # Shape: (batch_size, num_nodes, out_channels)

        # Prepare for message passing
        h_i = h.unsqueeze(2).expand(-1, -1, num_nodes, -1)  # (batch_size, num_nodes, num_nodes, out_channels)
        h_j = h.unsqueeze(1).expand(-1, num_nodes, -1, -1)  # (batch_size, num_nodes, num_nodes, out_channels)

        # Compute attention scores
        edge_features = torch.cat([h_i, h_j], dim=-1)  # (batch_size, num_nodes, num_nodes, 2 * out_channels)
        attention_weights = self.leakyrelu(self.attention(edge_features))  # (batch_size, num_nodes, num_nodes, 1)
        attention_weights = attention_weights.squeeze(-1)  # (batch_size, num_nodes, num_nodes)

        # Apply softmax to normalize the attention weights across edges
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Gate mechanism
        gate = torch.sigmoid(self.U(h_i) + self.V(h_j))  # (batch_size, num_nodes, num_nodes, out_channels)

        # Message passing: compute node updates
        node_features = gate * attention_weights.unsqueeze(-1) * h_j  # (batch_size, num_nodes, num_nodes, out_channels)

        # Apply adjacency matrix for message aggregation
        # Since adj is (num_nodes, num_nodes), we use batch matrix multiplication for each batch.
        aggr_out = torch.einsum("bijf,jk->bif", node_features, adj)  # Apply 2D adjacency matrix for aggregation

        # Apply layer normalization
        aggr_out = self.layer_norm(aggr_out)  # (batch_size, num_nodes, out_channels)

        return aggr_out


class GateGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout=0.1):
        super(GateGCN, self).__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(GatedGraphConv(in_channels, hidden_channels, dropout))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GatedGraphConv(hidden_channels, hidden_channels, dropout))

        # Output layer
        self.layers.append(GatedGraphConv(hidden_channels, out_channels, dropout))

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        """
        x: Tensor of shape (batch_size, num_nodes, in_channels)
        adj: Tensor of shape (num_nodes, num_nodes) -> Shared adjacency matrix for the whole batch
        """
        # Pass through Gated Graph Convolution layers
        for layer in self.layers[:-1]:
            x_res = x  # Residual connection
            x = layer(x, adj)
            x = self.relu(x + x_res)  # Residual connection with ReLU activation
            x = self.dropout(x)

        # Output layer without ReLU and dropout
        x = self.layers[-1](x, adj)
        return x
