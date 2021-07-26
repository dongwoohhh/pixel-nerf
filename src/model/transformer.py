import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import util


class RadianceTransformer2(nn.Module):
    """
    Represents RadianceTransformer2
    """

    def __init__(
        self,
        d_q,
        d_k,
        n_dim,
        n_head,
        n_layer
    ):
        super(RadianceTransformer2, self).__init__()

        self.n_layer = n_layer
        self.layers = []
        # Cls token for sigma prediction.
        #self.cls_token = nn.Parameter(torch.zeros(1, 1, n_dim))
        #nn.init.normal_(self.cls_token, std=0.02)
        # Linear Input latent projection
        self.linear1 = nn.Linear(d_k, n_dim)
        nn.init.constant_(self.linear1.bias, 0.0)
        nn.init.kaiming_normal_(self.linear1.weight, a=0, mode="fan_in")

        # Transformer for self-attention of src latent vector.
        for i in range(n_layer):
                self.layers.append(TransformerEncoderLayer(n_dim, n_head))

        self.layers = nn.ModuleList(self.layers)
        # Input slf_attn layer.
        self.attn_from_ref_to_src = MultiHeadAttentionLayer(n_query=d_q, n_key=n_dim, n_value=n_dim, n_dim=n_dim, n_head=n_head)

        self.layer_color = nn.Linear(n_dim, 3)
        nn.init.constant_(self.layer_color.bias, 0.0)
        nn.init.kaiming_normal_(self.layer_color.weight, a=0, mode="fan_in")
        
        self.layer_sigma = nn.Linear(n_dim, 1)
        nn.init.constant_(self.layer_sigma.bias, 0.0)
        nn.init.kaiming_normal_(self.layer_sigma.weight, a=0, mode="fan_in")

        self.activation = nn.ReLU()

    def forward(self, query, latent):
        out = self.linear1(latent)
        out = self.activation(out)
        #cls_token = self.cls_token.repeat(out.shape[0], 1, 1)
        #out = torch.cat([out, cls_token], dim=1)

        attn_prob_list = []
        for layer in self.layers:
            out = layer(out)
            attn_prob_list.append(layer.multi_head_attention_layer.attention_prob)
        attn_prob_list = torch.stack(attn_prob_list, dim=1)

        #out_token = out[:, -1, :]
        #out_latent = out[:, :-1, :]

        sigma = torch.max(out, dim = 1)[0]
        sigma = self.layer_sigma(sigma)
        #sigma = self.layer_sigma(self.activation(out_token))

        #out = self.attn_from_ref_to_src(query=query, key=out_latent, value=out_latent)
        #color = self.layer_color(out)
        #color = self.forward_attention_to_source(query=query, key=out_latent, value=out_latent)
        color = self.forward_attention_to_source(query=query, key=out, value=out)

        return color, sigma, out, attn_prob_list

    def forward_attention_to_source(self, query, key, value):
        out = self.attn_from_ref_to_src(query, key, value)
        color = self.layer_color(self.activation(out))

        return color


class TransformerEncoderLayer(nn.Module):
    def __init__(self, n_dim, n_head):
        super(TransformerEncoderLayer, self).__init__()
        self.multi_head_attention_layer = MultiHeadAttentionLayer(n_dim, n_dim, n_dim, n_dim, n_head=n_head)
        self.position_wise_feed_forward_layer = PositionWiseFeedForwardLayer(n_dim, n_dim, n_dim)

        self.norm_layer1 = nn.LayerNorm(n_dim)
        self.norm_layer2 = nn.LayerNorm(n_dim)

    def forward(self, x):
        out = self.multi_head_attention_layer(x, x, x) + x
        out = self.norm_layer1(out)

        out = self.position_wise_feed_forward_layer(out) + out
        out = self.norm_layer2(out)

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
        nn.init.constant_(self.query_fc_layer.bias, 0.0)
        nn.init.kaiming_normal_(self.query_fc_layer.weight, a=0, mode="fan_in")
        
        self.key_fc_layer = nn.Linear(n_key, n_dim*n_head)
        nn.init.constant_(self.key_fc_layer.bias, 0.0)
        nn.init.kaiming_normal_(self.key_fc_layer.weight, a=0, mode="fan_in")
        
        self.value_fc_layer = nn.Linear(n_value, n_dim*n_head)
        nn.init.constant_(self.value_fc_layer.bias, 0.0)
        nn.init.kaiming_normal_(self.value_fc_layer.weight, a=0, mode="fan_in")
        
        self.fc_layer = nn.Linear(n_dim*n_head, n_dim)
        nn.init.constant_(self.fc_layer.bias, 0.0)
        nn.init.kaiming_normal_(self.fc_layer.weight, a=0, mode="fan_in")

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

        self.attention_prob = F.softmax(attention_score, dim=-1)  # (B, N_ref, N_src)
        out = torch.matmul(self.attention_prob, value)  # (B, N_ref, D)

        return out

class PositionWiseFeedForwardLayer(nn.Module):
    def __init__(self, n_dim_in, n_dim1, n_dim2):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.first_fc_layer = nn.Linear(n_dim_in, n_dim1)
        nn.init.constant_(self.first_fc_layer.bias, 0.0)
        nn.init.kaiming_normal_(self.first_fc_layer.weight, a=0, mode="fan_in")
        
        self.second_fc_layer = nn.Linear(n_dim1, n_dim2)
        nn.init.constant_(self.second_fc_layer.bias, 0.0)
        nn.init.kaiming_normal_(self.second_fc_layer.weight, a=0, mode="fan_in")

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        out = self.first_fc_layer(x)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.second_fc_layer(out)

        return out
