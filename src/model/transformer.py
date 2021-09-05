import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import util

class RadianceTransformer3(nn.Module):
    """
    Represents RadianceTransformer3
    """
    def __init__(
        self,
        d_input,
        d_code,
        d_view,
        d_latent,
        d_color,
        n_latent,
        n_layer,
        n_head,
        n_iteration,
    ):
        super(RadianceTransformer3, self).__init__()
        self.n_layer = n_layer
        self.n_iteration = n_iteration
        self.d_code = d_code

        # Learnable parameters for latent array.
        self.latent_init = nn.Parameter(torch.zeros(1, 1, d_latent))
        nn.init.normal_(self.latent_init, std=0.02)

        # Linear view_direction and input to latent dimension..
        self.linear_code = nn.Linear(d_code, d_latent)
        self.linear_input = nn.Linear(d_input, d_latent)

        # Cross-atention from latent to input.
        self.cross_attn_encoder = MultiHeadAttentionLayer(
            d_query=d_latent, d_key=d_latent, d_value=d_latent, n_dim=d_latent, n_head=n_latent,
            aggregate=None
        )
        # Latent transformer.
        self.latent_transformer = []
        for i in range(n_layer):
            self.latent_transformer.append(
                TransformerEncoderLayer(n_dim=d_latent, n_head=n_head)
            )
        self.latent_transformer = nn.ModuleList(self.latent_transformer)
        # Decoding latent to input space.
        self.cross_attn_decoder = MultiHeadAttentionLayer(
            d_query=d_latent, d_key=d_latent, d_value=d_latent, n_dim=d_latent, n_head=1,
            aggregate=None
        )
        # Update latent.
        self.linear_latent = nn.Linear(n_latent*d_latent, d_latent)
        # Color decoder.
        self.linear_color_query = nn.Linear(d_view, d_color//2)
        self.decoder_color = MultiHeadAttentionLayer(
            d_query=d_color//2, d_key=d_latent, d_value=d_latent, n_dim=d_color, n_head=1,
        )
        self.layer_color1 = nn.Linear(d_color, d_color//2)
        self.layer_color2 = nn.Linear(d_color//2, 3)
        # Sigma decoder.
        self.layer_sigma1 = nn.Linear(d_latent, d_latent//2)
        self.layer_sigma2 = nn.Linear(d_latent//2, 1)

        self.activation = nn.ReLU()        
        
    def forward(self, input, view_dir):
        torch.autograd.set_detect_anomaly(True)
        n_batch = input.shape[0]
        # Split input and code.
        code = input[:, :, -self.d_code:]
        byte = input[:, :, :-self.d_code]
        # Input and code dimension to input dimension.
        byte = self.linear_input(byte)
        byte += self.linear_code(code)
        byte = self.activation(byte)
        # Latent init.
        latent_init = self.latent_init.repeat(n_batch, 1, 1)
        latent = latent_init

        sigma_multi = []
        # Iterative attend input.
        for i in range(self.n_iteration):
            # Cross-attention encoder.
            latent_init_identity = latent_init
            
            latent = self.cross_attn_encoder(query=latent_init, key=byte, value=byte)
            latent = latent.squeeze(1)
            # Transformer.
            for layer in self.latent_transformer:
                latent = layer(latent)
            # Cross_attention decoder.
            byte_identity = byte
            
            byte_update = self.cross_attn_decoder(query=byte, key=latent, value=latent)
            byte_update = byte_update.squeeze(2)
            
            # Update byte.
            byte = byte_identity + byte_update
            byte = self.activation(byte)
            
            # Update latent.
            latent_update = self.linear_latent(latent.reshape(n_batch, 1, -1))
            latent_init = latent_init_identity + latent_update
            latent_init = self.activation(latent_init)

            # Decode sigma.
            sigma_i = torch.max(byte, dim=1, keepdim=True)[0]
            sigma_i = self.layer_sigma1(sigma_i)
            sigma_i = self.layer_sigma2(self.activation(sigma_i))
            sigma_multi.append(sigma_i)
        
        sigma_multi = torch.stack(sigma_multi, dim=1)

        # Decode color.
        view_dir = self.linear_color_query(view_dir)
        color = self.decoder_color(query=view_dir, key=byte, value=byte)
        color = self.layer_color1(color)
        color = self.layer_color2(self.activation(color))
    
        return color, sigma_multi
    @classmethod
    def from_conf(cls, conf, d_input, d_code, d_view):
        return cls(
            d_input,
            d_code,
            d_view,
            d_latent=conf.get_int("d_latent", 128),
            d_color=conf.get_int("d_color", 64),
            n_latent=conf.get_int("n_latent", 3),
            n_layer=conf.get_int("n_layer", 4),
            n_head=conf.get_int("n_head", 4),
            n_iteration=conf.get_int("n_iteration", 4),
        )

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_query, d_key, d_value, d_latent, n_head):
        super(CrossAttentionLayer, self).__init__()
        self.d_latent = d_latent
        self.n_head = n_head
        # cross attention.
        self.query_fc_layer = nn.Linear(d_query, d_latent*n_head)
        self.key_fc_layer = nn.Linear(d_key, d_latent*n_head)
        self.value_fc_layer = nn.Linear(d_value, d_latent*n_head)
        # multi embedding attention.
        self.query_fc_layer2 = nn.Linear(d_latent, d_latent)
        self.key_fc_layer2 = nn.Linear(d_latent, d_latent)
        self.value_fc_layer2 = nn.Linear(d_latent, d_latent)

        self.activation = nn.ReLU()

    def forward(self, query, key, value, mask=None):
        n_batch = query.shape[0]
        query_init = query

        def transform(x, fc_layer):
            out = fc_layer(x)  # (B, N, H*D)
            out = out.view(n_batch, -1, self.n_head, self.d_latent)  # (B, N, H, D)
            out = out.transpose(1, 2)  # (B, H, N, D)
            
            return out
        # Compute cross attention.
        query = transform(query, self.query_fc_layer)  # (B, H, 1, D)
        key = transform(key, self.key_fc_layer)  # (B, H, N, D)
        value = transform(value, self.value_fc_layer)  # (B, H, N, D)
       
        out1 = self.calculate_attention(query, key, value)
        out1 = out1.transpose(1, 2)  # (B, 1, H, D)
        out1 = self.activation(out1)
        
        # Multi-embedding attetion.
        query2 = self.query_fc_layer2(out1)  # (B, 1, H, D)
        key2 = self.key_fc_layer2(out1)  # (B, 1, H, D)
        value2 = self.value_fc_layer2(out1)  # (B, 1, H, D)

        out2 = self.calculate_attention2(query2, key2, value2)  # (B, 1, D)
        out2 =self.activation(out2)

        out2 += query_init
        out1 = out1.squeeze(1)

        return out1, out2

    def calculate_attention(self, query, key, value):
        n_dim_key = key.size(-1)
        attention_score = torch.matmul(query, key.transpose(-2, -1))  # Q x K.T
        attention_score = attention_score / math.sqrt(n_dim_key)

        attention_prob = F.softmax(attention_score, dim=-1)  # (B, N_ref, N_src)
        out = torch.matmul(attention_prob, value)  # (B, N_ref, D)

        return out

    def calculate_attention2(self, query, key, value):
        n_dim_key = key.size(-1)
        attention_score = torch.matmul(query.unsqueeze(3), key.unsqueeze(4))
        attention_score = attention_score.squeeze(4)
        attention_score = attention_score / math.sqrt(n_dim_key)

        attention_prob = F.softmax(attention_score, dim=-1)  # (B, 1, H)
        
        out = torch.mul(attention_prob, value)  # (B, 1, D)
        out = torch.sum(out, dim=2)

        return out


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

    def forward(self, query, latent):
        out = self.linear1(latent)
        out = self.activation(out)

        for layer in self.layers:
            out = layer(out)

        sigma = torch.max(out, dim = 1)[0]
        sigma = self.layer_sigma(self.activation(sigma))
        out = self.attn_from_ref_to_src(query=query, key=out, value=out)

        color = self.layer_color(self.activation(out))

        return color, sigma


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

        for i in range(n_layer):
            self.layers.append(TransformerEncoderLayer(n_dim, n_head))
        self.layers = nn.ModuleList(self.layers)

        self.layer_color = nn.Linear(n_dim, 3)

    def forward(self, query, key, value):
        out = self.slf_attn(query, key, value)  # (B, N_ref, D)
        
        for layer in self.layers:
            out = layer(out)
        
        out = self.layer_color(out)  # (B, N_ref, D)

        return out

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
    def __init__(self, d_query, d_key, d_value, n_dim, n_head, aggregate="linear"):
        super(MultiHeadAttentionLayer, self).__init__()
        self.n_dim = n_dim
        self.n_head = n_head
        self.aggregate = aggregate

        self.query_fc_layer = nn.Linear(d_query, n_dim*n_head)
        self.key_fc_layer = nn.Linear(d_key, n_dim*n_head)
        self.value_fc_layer = nn.Linear(d_value, n_dim*n_head)
        if aggregate == "linear":
            self.fc_layer = nn.Linear(n_dim*n_head, n_dim)

    def forward(self, query, key, value):
        
        # query's shape: (B, N_query, 2)
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
        out = out.transpose(1, 2)  # (B, N_query, H, D)
        
        if self.aggregate == "linear":
            out = out.contiguous().view(n_batch, -1, self.n_head*self.n_dim)
            out = self.fc_layer(out)  # (B, N_query, D)

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
