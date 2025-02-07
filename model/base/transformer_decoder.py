# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from model.base.cyc_attn_layer import CyCAttention

import math
import numpy as np
def get_gauss(mu, sigma):
    gauss = lambda x: (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return gauss


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
    
    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CycAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = CyCAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory_key, memory_value, mask,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     value_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory_key, pos),
                                   value=self.with_pos_embed(memory_value, value_pos),
                                   mask=mask,
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return tgt

    def forward_pre(self, tgt, memory_key, memory_value, mask,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    value_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory_key, pos),
                                   value=self.with_pos_embed(memory_value, value_pos),
                                   mask=mask,
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory_key, memory_value, mask,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                value_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory_key, memory_value, mask, memory_mask,
                                    memory_key_padding_mask, pos, query_pos, value_pos)
        return self.forward_post(tgt, memory_key, memory_value, mask, memory_mask,
                                 memory_key_padding_mask, pos, query_pos, value_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory_key, memory_value,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     value_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory_key, pos),
                                   value=self.with_pos_embed(memory_value, value_pos), attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return tgt

    def forward_pre(self, tgt, memory_key, memory_value,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    value_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory_key, pos),
                                   value=self.with_pos_embed(memory_value, value_pos), attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory_key, memory_value,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                value_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory_key, memory_value, memory_mask,
                                    memory_key_padding_mask, pos, query_pos, value_pos)
        return self.forward_post(tgt, memory_key, memory_value, memory_mask,
                                 memory_key_padding_mask, pos, query_pos, value_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class transformer_decoder(nn.Module):

  """ Transformer decoder to get point query"""
  def __init__(self, args, num_queries, hidden_dim, dim_feedforward, nheads=4, num_layers=1, pre_norm=False):
    super().__init__()
    N_steps = hidden_dim // 2
    self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

    # define Transformer decoder here
    self.num_heads = nheads
    self.num_layers = num_layers
    self.transformer_self_attention_layers = nn.ModuleList()
    # self.transformer_self_attention_layers_0 = nn.ModuleList()
    self.transformer_cross_attention_layers = nn.ModuleList()
    self.transformer_cross_attention_layers_0 = nn.ModuleList()
    self.transformer_ffn_layers = nn.ModuleList()

    for _ in range(self.num_layers):
        self.transformer_self_attention_layers.append(
            SelfAttentionLayer(
                d_model=hidden_dim,
                nhead=nheads,
                dropout=0.0,
                normalize_before=pre_norm,
            )
        )
        self.transformer_cross_attention_layers_0.append(                                                                                                                    
            CycAttentionLayer(                                                                                                                                             
                d_model=hidden_dim,                                                                                                                                          
                nhead=nheads,                                                                                                                                                
                dropout=0.0,                                                                                                                                                 
                normalize_before=pre_norm,                                                                                                                                   
            )                                                                                                                                                                
        )
        self.transformer_cross_attention_layers.append(
            CycAttentionLayer(
                d_model=hidden_dim,
                nhead=nheads,
                dropout=0.0,
                normalize_before=pre_norm,
            )
        )    
    
        self.num_queries = num_queries

        self.supp_q_feat = nn.Embedding(num_queries, hidden_dim)
        self.q_embed = nn.Embedding(num_queries, hidden_dim)  
        
        
  def forward(self, input, stage):
    if stage == 1:
        return self.stage1(*input)
    elif stage == 2:
        return self.stage2(*input)       


  def stage1(self, x, x_s, support_mask, nshot):
    bs, C, H, W = x.shape
                                                                                                                              
    pos_x_s = self.pe_layer(x_s, None).flatten(2).to(x_s.device).permute(2, 0, 1).reshape(H*W, bs, nshot, C)                                                                                            
    src_x_s = x_s.flatten(2).permute(2, 0, 1).reshape(H*W, bs, nshot, C)                                                                                                                           
                                                                                                                                                                   
    q_supp_out = self.supp_q_feat.weight.unsqueeze(1).repeat(1, bs, 1)   
    query_pe = self.q_embed.weight.unsqueeze(1).repeat(1, bs, 1)      
    support_mask = support_mask.flatten(1).reshape(bs, nshot, H*W)         

    # attention: cross-attention first
    for s_idx in range(nshot):
        output = self.transformer_cross_attention_layers_0[0](                                                                                                               
            q_supp_out, src_x_s[:, :, s_idx], src_x_s[:, :, s_idx], support_mask[:, s_idx],                                                                                                                                        
            memory_mask=None,                                                                                                                                                
            memory_key_padding_mask=None,  # here we do not apply masking on padded region                                                                                   
            pos=pos_x_s[:, :, s_idx], query_pos=query_pe                                                                                                                            
        )                                                                                                                                                          
        
    return output.permute(1, 0, 2)
  
  
  def stage2(self, x, query_mask, output):
    bs, C, H, W = x.shape
    pos_x = self.pe_layer(x, None).flatten(2).to(x.device).permute(2, 0, 1)                                                                                                  
    src_x = x.flatten(2).permute(2, 0, 1) 
    
    query_pe = self.q_embed.weight.unsqueeze(1).repeat(1, bs, 1)  
    query_mask = query_mask.flatten(1)    
    
    output = self.transformer_cross_attention_layers[0](
        output, src_x, src_x, query_mask,
        memory_mask=None,
        memory_key_padding_mask=None,  # here we do not apply masking on padded region
        pos=pos_x, query_pos=query_pe, value_pos=None
    )
        
    output = self.transformer_self_attention_layers[0](
        output, tgt_mask=None,
        tgt_key_padding_mask=None,
        query_pos=query_pe
    )
    
    return output.permute(1, 0, 2)