# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DiT: https://github.com/facebookresearch/DiT
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final
from timm.models.vision_transformer import Attention, Mlp, RmsNorm, use_fused_attn

from .layers import (
    FFWRelativeCrossAttentionModule,
)
from .position_encodings import RotaryPositionEncoding3D
#################################################################################
#               Embedding Layers for Timesteps and Condition Inptus             #
#################################################################################
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=torch.bfloat16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.dtype = dtype

    def timestep_embedding(self, t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) *
                          torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(self.dtype)

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


#################################################################################
#                          Cross Attention Layers                               #
#################################################################################
class CrossAttention(nn.Module):
    """
    A cross-attention layer with flash attention.
    """
    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0,
        proj_drop: float = 0,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = use_fused_attn()

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, c: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, N, C = x.shape
        _, L, _ = c.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(c).reshape(B, L, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # Prepare attn mask (B, L) to mask the conditioion
        if mask is not None:
            mask = mask.reshape(B, 1, 1, L)
            mask = mask.expand(-1, -1, N, -1)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(query=q,
                                               key=k,
                                               value=v,
                                               dropout_p=self.attn_drop.p if self.training else 0.,
                                               attn_mask=mask)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if mask is not None:
                attn = attn.masked_fill_(mask.logical_not(), float('-inf'))
            attn = attn.softmax(dim=-1)
            if self.attn_drop.p > 0:
                attn = self.attn_drop(attn)
            x = attn @ v

        x = x.permute(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        if self.proj_drop.p > 0:
            x = self.proj_drop(x)
        return x


class JointTokenEncoder(nn.Module):
    def __init__(self, dim: int, num_arms: int):
        super().__init__()
        self.dim = dim
        self.num_arms = num_arms
        C = dim

        self.expand = nn.Sequential(
            nn.Linear(C, 1536),
            nn.GELU(),
            nn.Linear(1536, 1536),
        )
        
        self.gamma_embed = nn.Embedding(num_arms, 1536)
        self.beta_embed  = nn.Embedding(num_arms, 1536)

        # self.film = nn.Linear(C, 2 * C)

    def forward(self, g):  # g: [B,T,C]

        B, T, C = g.shape

        base = self.expand(g)  # [B,T,1536]
        q = base.unsqueeze(2).expand(B, T, self.num_arms, 1536).contiguous()  # [B,T,J,C_ds]
        # q = base.reshape(B, T, self.num_arms, C).contiguous()  # [B,T,J,C_ds]

        joint_ids = torch.arange(self.num_arms, device=g.device)
        gamma = self.gamma_embed(joint_ids)  # [J, C]
        beta  = self.beta_embed(joint_ids)   # [J, C]

        q = q * (1 + gamma.view(1,1,self.num_arms, 1536)) + beta.view(1,1,self.num_arms, 1536)
        return q   # [B,T,J,1536]

class JointTokenDecoder(nn.Module):
    """
    q [B,T,J,C] -> g [B,T,C]
    """
    def __init__(self, 
                 dim: int, num_arms: int, 
                 eps: float = 1e-4):
        super().__init__()
        self.dim = dim
        self.num_arms = num_arms
        C = dim
        self.eps = eps

        self.agg = nn.Linear(1536*num_arms, 1536)
        
        self.post = nn.Sequential(
            nn.LayerNorm(1536),
            nn.Linear(1536, 1536),
            nn.GELU(),
            nn.Linear(1536, dim),
        )
    
              
        self.register_buffer("joint_ids", torch.arange(num_arms), persistent=False)

    def forward(self, q):  # q: [B,T,J,C]
        B, T, J, C = q.shape
        h_agg = F.gelu(self.agg(q.reshape(B,T,J*C)))   # [B,T,1,J*C_ds] -> [B,T,1,C]
        g = self.post(h_agg)                  # [B,T,C]
        return g

def get_eef_xyz_from_coords(coords: np.ndarray) -> np.ndarray:
    coord_left_gripper = (coords[:, 6] + coords[:, 7]) / 2
    coord_right_gripper = (coords[:, 14] + coords[:, 15]) / 2
    # return np.concatenate((coord_left_gripper, coord_right_gripper), axis=0)  # (3,)
    return  coord_left_gripper, coord_right_gripper

def get_gt_xyz(left_joint_angles, right_joint_angles, fk_engine) -> np.ndarray:
    B = left_joint_angles.shape[0]
    device = left_joint_angles.device
    dtype = left_joint_angles.dtype
    
    coords_list = []

    for b in range(B):
        l = left_joint_angles[b][0] 
        r = right_joint_angles[b][0] 

        if l.shape[0] == 6:
            l = torch.cat([l, l.new_zeros(2)], dim=0)
        # l = l.detach().cpu().numpy()
        l = l.detach().to(torch.float32).cpu().numpy()
    
        if r.shape[0] == 6:
            r = torch.cat([r, r.new_zeros(2)], dim=0)
        # r = r.detach().cpu().numpy()
        r = r.detach().to(torch.float32).cpu().numpy()

        angles_np = np.concatenate([l, r], axis=0)      # (16,)
        coords_np = fk_engine.fk(angles_np)             # (16,3) numpy
        coords_list.append(torch.from_numpy(coords_np)) # torch [16,3] (cpu)

    coords = torch.stack(coords_list, dim=0).to(device=device, dtype=dtype)  # [B,16,3]
    coord_left_gripper, coord_right_gripper = get_eef_xyz_from_coords(coords)
    coords = torch.cat([coords[:,0:6,:], coord_left_gripper.unsqueeze(1), coords[:,8:14,:], coord_right_gripper.unsqueeze(1)], dim=1) 
    return coords
    

#################################################################################
#                                 RDT Block                                     #
#################################################################################
class RDTBlock(nn.Module):
    """
    A RDT block with cross-attention conditioning.
    """

    def __init__(self, hidden_size, num_heads, enable_pcd, **block_kwargs):
        super().__init__()
        self.norm1 = RmsNorm(hidden_size, eps=1e-6)
        self.attn = Attention(dim=hidden_size,
                              num_heads=num_heads,
                              qkv_bias=True,
                              qk_norm=True,
                              norm_layer=RmsNorm,
                              **block_kwargs)
        self.cross_attn = CrossAttention(hidden_size,
                                         num_heads=num_heads,
                                         qkv_bias=True,
                                         qk_norm=True,
                                         norm_layer=RmsNorm,
                                         **block_kwargs)

        self.norm2 = RmsNorm(hidden_size, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.ffn = Mlp(in_features=hidden_size, hidden_features=hidden_size, act_layer=approx_gelu, drop=0)
        self.norm3 = RmsNorm(hidden_size, eps=1e-6)
        self.enable_pcd = enable_pcd
        if self.enable_pcd:
            self.hidden_size = hidden_size
            # New: cross_attn_pcd
            self.cross_attn_pcd = FFWRelativeCrossAttentionModule(
                embedding_dim=1536,
                num_attn_heads=8,
                num_layers=2,
                use_adaln=True
            )
            self.relative_pe_layer = RotaryPositionEncoding3D(1536)
            
            # New:Film module
            self.num_arms = 2  
        
            self.joint_encoder = JointTokenEncoder(dim=hidden_size, num_arms=self.num_arms)
            self.joint_decoder = JointTokenDecoder(dim=hidden_size, 
                                                    num_arms=self.num_arms)
            
            self.temb_proj_for_cross_3d = nn.Sequential(
                    nn.Linear(hidden_size, 1536)  
                )
            
            self.pcd_dino_mlp = nn.Linear(1024, 1536)
            # self.pcd_dino_mlp = nn.Linear(3, 1536)

            # self.pcd_dino_mlp = nn.Linear(2048, 1536)
    

    def forward(self, x, c, mask=None, pcd_output=None, fk_engine = None, left_arm_state = None, right_arm_state = None, temb=None):
        origin_x = x
        x = self.norm1(x)
        x = self.attn(x)
        x = x + origin_x

        origin_x = x
        x = self.norm2(x)
        norm_x = x
        x = self.cross_attn(norm_x, c, mask)

        if self.enable_pcd and (pcd_output is not None):
            pcd_coord = pcd_output["pcd_coord"]     # (B, N, 3)
            pcd_feat  = pcd_output["pcd_dino"]  # (B, N, D)
            pcd_batch_mask = pcd_output['pcd_batch_mask']
            
            key_padding_mask = ~pcd_batch_mask

            B, T, C = norm_x.shape # torch.Size([1, 67, 2048])
            J = self.num_arms
            joint_tokens = self.joint_encoder(norm_x)  # torch.Size([1, 67, 2, 1536])

            joint_coords = get_gt_xyz(left_arm_state, right_arm_state, fk_engine)
            # rel_q_pos_single = self.relative_pe_layer(joint_coords)               # [B, J, C_ds, 2]
            if joint_coords.device != x.device:
                joint_coords = joint_coords.to(x.device)
            rel_q_pos_single = self.relative_pe_layer(joint_coords[:, [7, 13], :])               # [B, J, C_ds, 2]
            rel_q_pos = rel_q_pos_single.unsqueeze(1).expand(B, T, J, 1536, 2)       # [B, T, J, C_ds, 2]
            # rel_q_pos = rel_q_pos.contiguous().view(B, T * J, self.C_ds, 2)
            rel_q_pos = rel_q_pos.reshape(B, T * J, 1536, 2)
            rel_v_pos = self.relative_pe_layer(pcd_coord)                     # [B, N, C_ds, 2]

            q = joint_tokens.reshape(B, T*J, 1536).transpose(0, 1).float() # [T*J, B, C_ds] torch.Size([134, 1, 1536])
            q_pos = rel_q_pos.float()                 
            v_pos = rel_v_pos                     
            ''''
              Finish
            '''
            # cross-attn
        
            pcd_feat = pcd_feat.to(self.pcd_dino_mlp.weight.dtype) # torch.Size([1, 946, 1024])
            # pcd_feat = pcd_coord.to(self.pcd_dino_mlp.weight.dtype)
            v = self.pcd_dino_mlp(pcd_feat)
            
            v =  v.transpose(0, 1)
            temb = temb.squeeze(1)
            temb = self.temb_proj_for_cross_3d(temb)
            action_output_multi = self.cross_attn_pcd(
                query=q,
                value=v,
                query_pos=q_pos,
                value_pos=v_pos,
                diff_ts=temb,
                pad_mask=key_padding_mask
            )[-1]
            
            action_output_multi = action_output_multi.transpose(0, 1).reshape(B, T, J, 1536)  # [B,T,J,C] torch.Size([134, 1, 1536])
            act_pcd_output = self.joint_decoder(action_output_multi)
        
            x = origin_x + x + act_pcd_output  # torch.Size([1, 67, 2048])
        
        else:
            x = origin_x + x 

        origin_x = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = x + origin_x

        return x



class FinalLayer(nn.Module):
    """
    The final layer of RDT.
    """

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = RmsNorm(hidden_size, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.ffn_final = Mlp(in_features=hidden_size,
                             hidden_features=hidden_size,
                             out_features=out_channels,
                             act_layer=approx_gelu,
                             drop=0)

    def forward(self, x):
        x = self.norm_final(x)
        x = self.ffn_final(x)
        return x


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    if not isinstance(pos, np.ndarray):
        pos = np.array(pos, dtype=np.float64)
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_nd_sincos_pos_embed_from_grid(embed_dim, grid_sizes):
    """
    embed_dim: output dimension for each position
    grid_sizes: the grids sizes in each dimension (K,).
    out: (grid_sizes[0], ..., grid_sizes[K-1], D)
    """
    num_sizes = len(grid_sizes)
    # For grid size of 1, we do not need to add any positional embedding
    num_valid_sizes = len([x for x in grid_sizes if x > 1])
    emb = np.zeros(grid_sizes + (embed_dim, ))
    # Uniformly divide the embedding dimension for each grid size
    dim_for_each_grid = embed_dim // num_valid_sizes
    # To make it even
    if dim_for_each_grid % 2 != 0:
        dim_for_each_grid -= 1
    valid_size_idx = 0
    for size_idx in range(num_sizes):
        grid_size = grid_sizes[size_idx]
        if grid_size <= 1:
            continue
        pos = np.arange(grid_size)
        posemb_shape = [1] * len(grid_sizes) + [dim_for_each_grid]
        posemb_shape[size_idx] = -1
        emb[..., valid_size_idx * dim_for_each_grid:(valid_size_idx + 1) * dim_for_each_grid] += \
            get_1d_sincos_pos_embed_from_grid(dim_for_each_grid, pos).reshape(posemb_shape)
        valid_size_idx += 1
    return emb


def get_multimodal_cond_pos_embed(embed_dim, mm_cond_lens: OrderedDict, embed_modality=True):
    """
    Generate position embeddings for multimodal conditions. 
    
    mm_cond_lens: an OrderedDict containing 
        (modality name, modality token length) pairs.
        For `"image"` modality, the value can be a multi-dimensional tuple.
        If the length < 0, it means there is no position embedding for the modality or grid.
    embed_modality: whether to embed the modality information. Default is True.
    """
    num_modalities = len(mm_cond_lens)
    modality_pos_embed = np.zeros((num_modalities, embed_dim))
    if embed_modality:
        # Get embeddings for various modalites
        # We put it in the first half
        modality_sincos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, torch.arange(num_modalities))
        modality_pos_embed[:, :embed_dim // 2] = modality_sincos_embed
        # The second half is for position embeddings
        pos_embed_dim = embed_dim // 2
    else:
        # The whole embedding is for position embeddings
        pos_embed_dim = embed_dim

    # Get embeddings for positions inside each modality
    c_pos_emb = np.zeros((0, embed_dim))
    for idx, (modality, cond_len) in enumerate(mm_cond_lens.items()):
        if modality == "image" and \
            (isinstance(cond_len, tuple) or isinstance(cond_len, list)):
            all_grid_sizes = tuple([abs(x) for x in cond_len])
            embed_grid_sizes = tuple([x if x > 0 else 1 for x in cond_len])
            cond_sincos_embed = get_nd_sincos_pos_embed_from_grid(pos_embed_dim, embed_grid_sizes)
            cond_pos_embed = np.zeros(all_grid_sizes + (embed_dim, ))
            cond_pos_embed[..., -pos_embed_dim:] += cond_sincos_embed
            cond_pos_embed = cond_pos_embed.reshape((-1, embed_dim))
        else:
            cond_sincos_embed = get_1d_sincos_pos_embed_from_grid(pos_embed_dim,
                                                                  torch.arange(cond_len if cond_len > 0 else 1))
            cond_pos_embed = np.zeros((abs(cond_len), embed_dim))
            cond_pos_embed[:, -pos_embed_dim:] += cond_sincos_embed
        cond_pos_embed += modality_pos_embed[idx]
        c_pos_emb = np.concatenate([c_pos_emb, cond_pos_embed], axis=0)

    return c_pos_emb
