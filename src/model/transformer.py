import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import util

class RadianceTransformer(nn.Module):
    """
    Represents RadianceTransformer
    """

    def __init__(
        self,
        d_q,
        d_k,
        n_dim,
        n_head,
        n_layer
    ):
        super(RadianceTransformer, self).__init__()
    
        self.n_layer = n_layer
        self.layers = []
        # Input slf_attn layer.
        self.slf_attn = MultiHeadAttentionLayer(n_query=d_q, n_key=d_k, n_value=d_k, n_dim=n_dim, n_head=n_head)
        """
        for i in range(n_layer):
            self.layers.append(TransformerEncoderLayer(n_dim, n_head))
        self.layers = nn.ModuleList(self.layers)
        """
        self.layer_color = nn.Linear(n_dim, 3)

    def forward(self, query, key, value):
        out = self.slf_attn(query, key, value)  # (B, N_ref, D)
        """
        for layer in self.layers:
            out = layer(out)
        """
        out = self.layer_color(out)  # (B, N_ref, D)

        return out

class TransformerEncoderLayer(nn.Module):
    def __init__(self, n_dim, n_head):
        super(TransformerEncoderLayer, self).__init__()
        self.multi_head_attention_layer = MultiHeadAttentionLayer(n_dim, n_dim, n_dim, n_dim, n_head=n_head)
        self.position_wise_feed_forward_layer = PositionWiseFeedForwardLayer(n_dim, n_dim, n_dim)
        self.residual_connection_layer = [ResidualConnectionLayer(n_dim), ResidualConnectionLayer(n_dim)]

        self.norm_layer1 = nn.LayerNorm(n_dim)
        self.norm_layer2 = nn.LayerNorm(n_dim)

    def forward(self, x):
        out = self.residual_connection_layer[0](x, lambda x: self.multi_head_attention_layer(x, x, x))
        out = self.residual_connection_layer[1](x, lambda x: self.position_wise_feed_forward_layer(x))

        return out

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, n_query, n_key, n_value, n_dim, n_head):
        super(MultiHeadAttentionLayer, self).__init__()
        self.n_dim = n_dim
        self.n_head = n_head

        self.query_fc_layer = [] #nn.Linear(2, n_dim)
        self.key_fc_layer = [] #nn.Linear(n_dim, n_dim)
        self.value_fc_layer = [] #nn.Linear(n_dim, n_dim)
        
        
        self.query_fc_layer = nn.Linear(n_query, n_dim*n_head)
        self.key_fc_layer = nn.Linear(n_key, n_dim*n_head)
        self.value_fc_layer = nn.Linear(n_value, n_dim*n_head)
        
        self.fc_layer = nn.Linear(n_dim*n_head, n_dim)

    def forward(self, query, key, value):
        
        # query's shape: (B, N_ref, 2)
        # key, value's shape: (B, N_src, d_k)
        n_batch = query.shape[0]

        def transform(x, fc_layer):
            out = fc_layer(x)  # (B, N, H*D)
            out = out.view(n_batch, -1, self.n_head, self.n_dim)  # (B, N, H, D)
            out = out.transpose(1, 2)  # (B, H, N, D)
            
            return out
        
        query = transform(query, self.query_fc_layer)
        key = transform(key, self.key_fc_layer)
        value = transform(value, self.value_fc_layer)

        out = self.calculate_attention(query, key, value)
        out = out.transpose(1,2)  # (B, N_ref, H, D)
        out = out.contiguous().view(n_batch, -1, self.n_head*self.n_dim)
        out = self.fc_layer(out)  # (B, N_ref, D)

        return out

    def calculate_attention(self, query, key, value):
        n_dim_key = key.size(-1)
        attention_score = torch.matmul(query, key.transpose(-2, -1))  # Q x K.T
        attention_score = attention_score / math.sqrt(n_dim_key)

        attention_prob = F.softmax(attention_score, dim=-1)  # (B, N_ref, N_src)
        out = torch.matmul(attention_prob, value)  # (B, N_ref, D)

        return out

class PositionWiseFeedForwardLayer(nn.Module):
    def __init__(self, n_dim_in, n_dim1, n_dim2):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.first_fc_layer = nn.Linear(n_dim_in, n_dim1)
        self.second_fc_layer = nn.Linear(n_dim1, n_dim2)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        out = self.first_fc_layer(x)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.second_fc_layer(out)

        return out

class ResidualConnectionLayer(nn.Module):
    def __init__(self, n_dim):
        super(ResidualConnectionLayer, self).__init__()
        self.norm_layer = nn.LayerNorm(n_dim)

    def forward(self, x, sub_layer):
        out = sub_layer(x) + x
        out = self.norm_layer(out)

        return out