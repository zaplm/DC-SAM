import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from einops import rearrange

try:
    from deformable_attn_layer import DeformableAttention
except:
    from layers import DeformableAttention


def grid_sample_1d(feats, grid, *args, **kwargs):
    # does 1d grid sample by reshaping it to 2d

    grid = rearrange(grid, '... -> ... 1 1')
    grid = F.pad(grid, (1, 0), value = 0.)
    feats = rearrange(feats, '... -> ... 1')
    out = F.grid_sample(feats, grid, **kwargs)
    return rearrange(out, '... 1 -> ...')

def normalize_grid(arange, dim = 1, out_dim = -1):
    # normalizes 1d sequence to range of -1 to 1
    n = arange.shape[-1]
    return 2.0 * arange / max(n - 1, 1) - 1.0

class DeformCrossAttn(DeformableAttention):
    def __init__(self,
                 hidden_dim: int,
                 num_heads: int,
                 dropout: float = 0.1):
        super(DeformCrossAttn, self).__init__(dim=hidden_dim, dim_head=hidden_dim // num_heads, heads=num_heads, dropout=dropout)
    
    def forward(self, x, memory, return_vgrid = False):
        heads, b, n, downsample_factor, device = self.heads, x.shape[0], x.shape[-1], self.downsample_factor, x.device

        # queries

        q = self.to_q(x)

        # calculate offsets - offset MLP shared across all groups

        group = lambda t: rearrange(t, 'b (g d) n -> (b g) d n', g = self.offset_groups)

        grouped_queries = group(q)
        offsets = self.to_offsets(grouped_queries)

        # calculate grid + offsets

        grid = torch.arange(offsets.shape[-1], device = device)
        vgrid = grid + offsets
        vgrid_scaled = normalize_grid(vgrid)

        kv_feats = grid_sample_1d(
            group(memory),
            vgrid_scaled,
        mode = 'bilinear', padding_mode = 'zeros', align_corners = False)

        kv_feats = rearrange(kv_feats, '(b g) d n -> b (g d) n', b = b)

        # derive key / values

        k, v = self.to_k(kv_feats), self.to_v(kv_feats)

        # scale queries

        q = q * self.scale

        # split out heads

        q, k, v = map(lambda t: rearrange(t, 'b (h d) n -> b h n d', h = heads), (q, k, v))

        # query / key similarity

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # relative positional bias

        seq_range = torch.arange(n, device = device)
        seq_scaled = normalize_grid(seq_range, dim = 0)
        rel_pos_bias = self.rel_pos_bias(seq_scaled, vgrid_scaled)
        sim = sim + rel_pos_bias

        # numerical stability

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()

        # attention

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        # aggregate and combine heads

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b (h d) n')
        out = self.to_out(out)

        if return_vgrid:
            return out, vgrid

        return out


class DeformableAttentionLayer(nn.TransformerDecoderLayer):
    def __init__(self,
                 hidden_dim: int,
                 num_heads: int,
                 dropout: float = 0.1
                 ):
        super(DeformableAttentionLayer, self).__init__(hidden_dim, num_heads)
        # self.self_attn = DeformableAttention(dim=hidden_dim, dim_head=hidden_dim // num_heads, heads=num_heads, dropout=dropout)
        self.multihead_attn = DeformCrossAttn(hidden_dim, num_heads, dropout)
    
    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            x = x + self._mha_block(self.norm2(x), memory)
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm2(x + self._mha_block(x, memory))
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.norm3(x + self._ff_block(x))

        return x
    
    # def _sa_block(self, x: Tensor,
    #               attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
    #     x = x.permute(1, 2, 0)
    #     x = self.self_attn(x).permute(2, 0, 1)
    #     return self.dropout1(x)
    
    def _mha_block(self, x: Tensor, mem: Tensor) -> Tensor:
        x, mem = x.permute(1, 2, 0), mem.permute(1, 2, 0)
        x = self.multihead_attn(x, mem).permute(2, 0, 1)
        return self.dropout2(x)
        

if __name__ == "__main__":
    layer = DeformableAttentionLayer(256, 8)
    x = torch.randn(10, 2, 256)
    memory = torch.randn(100, 2, 256)
    print(layer(x, memory).shape)