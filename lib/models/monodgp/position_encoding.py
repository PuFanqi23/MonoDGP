# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

from utils.misc import NestedTensor
import numpy as np


def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    """
    Args:
        pos: (N_query, 3)
        num_pos_feats:
        temperature:
    Returns:
        posemb: (N_query, num_feats * 3)
    """
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)     # (num_feats, )
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)   # (num_feats, )   [10000^(0/128), 10000^(0/128), 10000^(2/128), 10000^(2/128), ...]
    pos_x = pos[..., 0, None] / dim_t   # (N_query, num_feats)      num_feats:  [pos_x/10000^(0/128), pos_x/10000^(0/128), pos_x/10000^(2/128), pos_x/10000^(2/128), ...]
    pos_y = pos[..., 1, None] / dim_t   # (N_query, num_feats)      num_feats:  [pos_y/10000^(0/128), pos_y/10000^(0/128), pos_y/10000^(2/128), pos_y/10000^(2/128), ...]
    pos_z = pos[..., 2, None] / dim_t   # (N_query, num_feats)      num_feats:  [pos_z/10000^(0/128), pos_z/10000^(0/128), pos_z/10000^(2/128), pos_z/10000^(2/128), ...]

    # (N_query, num_feats/2, 2) --> (N_query, num_feats)
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)       # num_feats:  [sin(pos_x/10000^(0/128)), cos(pos_x/10000^(0/128)), sin(pos_x/10000^(2/128)), cos(pos_x/10000^(2/128)), ...]
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)       # num_feats:  [sin(pos_y/10000^(0/128)), cos(pos_y/10000^(0/128)), sin(pos_y/10000^(2/128)), cos(pos_y/10000^(2/128)), ...]
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)       # num_feats:  [sin(pos_z/10000^(0/128)), cos(pos_z/10000^(0/128)), sin(pos_z/10000^(2/128)), cos(pos_z/10000^(2/128)), ...]
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)   # (N_query, num_feats * 3)
    return posemb


class PositionEmbeddingCamRay(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.position_range = [1e-3, -60, -8.0, 60.0, 60, 8.0]
        self.LID = True
        self.depth_num = 81
        self.depth_start = 1e-3
        self.num_pos_feats = num_pos_feats
        self.normalize = normalize
        self.depth_dim_keep = 24

        self.position_dim = 3 * self.depth_num      # D*3 3:(x, y, z)
        self.depth_position_dim = 3 * self.depth_dim_keep  # K*3 3:(x, y, z)
        self.embed_dims = 256
        self.position_encoder = nn.Sequential(
                nn.Conv2d(self.position_dim, self.embed_dims, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )
        self.depth_position_encoder = nn.Sequential(
                nn.Conv2d(self.depth_position_dim, self.embed_dims, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )
        self.position_sine = PositionEmbeddingSine(num_pos_feats, normalize=True)
        
    def forward(self, tensor_list: NestedTensor, calibs=None, depth_map=None):
        """
        Args:
            img_feats: List[(B, C, H, W), ]
            img_metas:
            masks: (B, H, W)
            depth_map: (B, H, W)
            depth_map_mask: (B, H, W)
        Returns:
            coords_position_embeding: (B, embed_dims, H, W)
            coords_mask: (B, H, W)
        """

        img_feats = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        #assert depth_map is not None

        B, C, H, W = img_feats.shape
        coords_h = torch.arange(H, device=img_feats[0].device).float()     # (H, )
        coords_w = torch.arange(W, device=img_feats[0].device).float()     # (W, )

        if self.LID:
            # (D, )
            index = torch.arange(start=0, end=self.depth_num, step=1, device=img_feats[0].device).float()
            index_1 = index + 1
            bin_size = (self.position_range[3] - self.depth_start) / (self.depth_num * (1 + self.depth_num))
            coords_d = self.depth_start + bin_size * index * index_1
        else:
            index = torch.arange(start=0, end=self.depth_num, step=1, device=img_feats[0].device).float()
            bin_size = (self.position_range[3] - self.depth_start) / self.depth_num
            coords_d = self.depth_start + bin_size * index
        
        D = coords_d.shape[0]


        # (3, W, H, D)  --> (W, H, D, 3)    3: (u, v, d)
        coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d])).permute(1, 2, 3, 0)
        #coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)      # (W, H, D, 4)    4: (du, dv, d, 1)
        # if self.normalize:
        #     eps = 1e-6
        #     coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)      # (W, H, D, 3)    3: (du, dv, d)
        
        # (1, W, H, D, 3) --> (B, W, H, D, 3)
        coords = coords.view(1, W, H, D, 3).repeat(B, 1, 1, 1, 1)

        # (B, W, H, D, 3) --> (B, D, 3, H, W) --> (B, D*3, H, W)
        coords3d = coords.permute(0, 3, 4, 2, 1).contiguous().view(B, -1, H, W)
        # 3D position embedding(PE)
        coords_position_embeding = self.position_encoder(coords3d)      # (B, embed_dims, H, W)

        if depth_map is not None:
            depth_map = F.interpolate(depth_map, size=img_feats.shape[2:], mode='bilinear', align_corners=True)
            depth_score = F.softmax(depth_map, dim=1)
            _, keep_indices = torch.topk(depth_score, k=self.depth_dim_keep, dim=1)
            # (B, W, H, K)
            keep_indices = keep_indices.permute(0, 3, 2, 1)
            keep_indices = keep_indices[..., None].repeat(1, 1, 1, 1, 3)
            coords = torch.gather(coords, dim=3, index=keep_indices)
            D = self.depth_dim_keep

            if calibs is not None:
                img2lidar = np.linalg.inv(calibs[:,:,:3].cpu())
                img2lidar = coords.new_tensor(img2lidar) # (B, 3, 3)
                # (B, W, H, D, 3, 3)
                img2lidar = img2lidar.view(B, 1, 1, 1, 3, 3).repeat(1, W, H, D, 1, 1)
                coords = coords.view(B, W, H, D, 3, 1)
                coords3d = torch.matmul(img2lidar, coords).squeeze(-1)
                
                rays = F.normalize(coords3d[..., :3], dim=-1)
                rays = rays.permute(0, 3, 4, 2, 1).contiguous().view(B, -1, H, W)
                rays_position_embeding = self.depth_position_encoder(rays)      # (B, embed_dims, H, W)

                return rays_position_embeding + self.position_sine(tensor_list)
            
            coords = F.normalize(coords, dim=-1)
            # (B, W, H, D, 3) --> (B, D, 3, H, W) --> (B, D*3, H, W)
            coords3d = coords.permute(0, 3, 4, 2, 1).contiguous().view(B, -1, H, W)
            # 3D position embedding(PE)
            coords_position_embeding = self.depth_position_encoder(coords3d)      # (B, embed_dims, H, W)

        return coords_position_embeding + self.position_sine(tensor_list)

    
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

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device) / w * 49
        j = torch.arange(h, device=x.device) / h * 49
        x_emb = self.get_embed(i, self.col_embed)
        y_emb = self.get_embed(j, self.row_embed)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos

    def get_embed(self, coord, embed):
        floor_coord = coord.floor()
        delta = (coord - floor_coord).unsqueeze(-1)
        floor_coord = floor_coord.long()
        ceil_coord = (floor_coord + 1).clamp(max=49)
        return embed(floor_coord) * (1 - delta) + embed(ceil_coord) * delta


def build_position_encoding(cfg):
    N_steps = cfg['hidden_dim'] // 2
    #cfg['position_embedding'] = 'pe3d'
    if cfg['position_embedding'] in ('v1', 'pe3d'):
        position_embedding = PositionEmbeddingCamRay(N_steps, normalize=True)
    elif cfg['position_embedding'] in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif cfg['position_embedding'] in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {cfg['position_embedding']}")

    return position_embedding
