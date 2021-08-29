import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class Perceiver(nn.Module):
    """
    Represents Perceiver
    """

    def __init__(
        self,
        d_input,
        d_latent,
        d_color,
        n_latent,
        n_head,
        n_layer,
        n_repeats=None,
    ):
        super(Perceiver, self).__init__()

        self.latent_init = nn.Parameter(torch.zeros(1, n_latent, d_latent))
        nn.init.normal_(self.latent_init, std=0.02)

        self.n_layer = n_layer

        # Cross-attention layer.
        self.cross_attn1 = MultiHeadAttentionLayer(n_query=d_latent, n_key=d_input, n_value=d_input, n_dim=d_latent, n_head=1)
        self.cross_attn2 = MultiHeadAttentionLayer(n_query=d_latent, n_key=d_input, n_value=d_input, n_dim=d_latent, n_head=1)

        # Transformer.
        self.layers = []
        for i in range(n_layer):
            self.layers.append(TransformerEncoderLayer(d_latent, n_head))
        self.layers = nn.ModuleList(self.layers)

        # Decoder.
        d_decoder = 32
        self.decoder_color = MultiHeadAttentionLayer(n_query=d_color, n_key=d_latent, n_value=d_latent, n_dim=d_decoder, n_head=1)

        self.layer_color = nn.Linear(d_decoder, 3)
        self.layer_sigma = nn.Linear(d_latent, 1)


        self.activation = nn.ReLU()
        
    def forward(self, input, query_color, mask):
        byte = input
        # Cross-attention 1.
        latent = self.latent_init.repeat(byte.shape[0], 1, 1)
        latent = self.cross_attn1(query=latent, key=byte, value=byte, mask=mask)

        # Latent transformer.
        for i, layer in enumerate(self.layers):
            latent = layer(latent, mask=None)
        
        latent = self.cross_attn2(query=latent, key=byte, value=byte, mask=mask)
        attn_prob_list = torch.stack([self.cross_attn2.attention_prob])

        # Latent transformer.
        for i, layer in enumerate(self.layers):
            latent = layer(latent, mask=None)

        sigma = torch.max(latent, dim=1)[0]
        sigma = self.layer_sigma(self.activation(sigma))


        color = self.forward_attention_to_source(query=query_color, key=latent, value=latent)

        return color, sigma, latent, attn_prob_list

    def forward_attention_to_source(self, query, key, value):
        out = self.decoder_color(query, key, value)
        color = self.layer_color(self.activation(out))

        return color

    

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

        # Transformer for self-attention of src latent vector.
        for i in range(n_layer):
            self.layers.append(TransformerEncoderLayer(n_dim, n_head))
        self.layers = nn.ModuleList(self.layers)

        # Input slf_attn layer.
        self.attn_from_ref_to_src = MultiHeadAttentionLayer(n_query=d_q, n_key=n_dim, n_value=n_dim, n_dim=n_dim, n_head=n_head)

        self.layer_color = nn.Linear(n_dim, 3)        
        self.layer_sigma = nn.Linear(n_dim, 1)

        self.activation = nn.ReLU()

    def forward(self, query, latent, mask=None):
        out = self.linear1(latent)
        out = self.activation(out)
        #cls_token = self.cls_token.repeat(out.shape[0], 1, 1)
        #out = torch.cat([out, cls_token], dim=1)

        attn_prob_list = []
        for i, layer in enumerate(self.layers):
            if i == 0:
                out = layer(out, mask)
            else:
                out = layer(out, mask=None)
            attn_prob_list.append(layer.multi_head_attention_layer.attention_prob)
        attn_prob_list = torch.stack(attn_prob_list, dim=1)

        #out_token = out[:, -1, :]
        #out_latent = out[:, :-1, :]

        sigma = torch.max(out, dim = 1)[0]
        sigma = self.layer_sigma(self.activation(sigma))
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

    def forward(self, x, mask):
        out = self.multi_head_attention_layer(x, x, x, mask) + x
        out = self.norm_layer1(out)

        out = self.position_wise_feed_forward_layer(out) + out
        out = self.norm_layer2(out)

        return out

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, n_query, n_key, n_value, n_dim, n_head):
        super(MultiHeadAttentionLayer, self).__init__()
        self.n_dim = n_dim
        self.n_head = n_head

        self.register_buffer("inf", torch.tensor([float('inf')]), persistent=True)

        self.query_fc_layer = [] #nn.Linear(2, n_dim)
        self.key_fc_layer = [] #nn.Linear(n_dim, n_dim)
        self.value_fc_layer = [] #nn.Linear(n_dim, n_dim)

        self.query_fc_layer = nn.Linear(n_query, n_dim*n_head)
        self.key_fc_layer = nn.Linear(n_key, n_dim*n_head)
        self.value_fc_layer = nn.Linear(n_value, n_dim*n_head)

        self.fc_layer = nn.Linear(n_dim*n_head, n_dim)

        self.activation = nn.ReLU()

    def forward(self, query, key, value, mask=None):
        
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

        out = self.calculate_attention(query, key, value, mask)
        out = out.transpose(1,2)  # (B, N_ref, H, D)
        out = out.contiguous().view(n_batch, -1, self.n_head*self.n_dim)
        out = self.fc_layer(self.activation(out))  # (B, N_ref, D)

        return out

    def calculate_attention(self, query, key, value, mask):
        n_dim_key = key.size(-1)
        attention_score = torch.matmul(query, key.transpose(-2, -1))  # Q x K.T
        attention_score = attention_score / math.sqrt(n_dim_key)

        # Mask view which has occlusion.
        if mask is not None:
            mask = mask[:, None].repeat(1, self.n_head, 1, 1)
            mask_inf = torch.where(
                mask, torch.zeros_like(attention_score), self.inf*torch.ones_like(attention_score)
            )
            attention_score = attention_score - mask_inf

        self.attention_prob = F.softmax(attention_score, dim=-1)  # (B, N_ref, N_src)
        out = torch.matmul(self.attention_prob, value)  # (B, N_ref, D)

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
