import torch
import torch.nn as nn
import math
import numpy as np

from .balancer import Balancer
from .focalloss import FocalLoss

from scipy.ndimage import gaussian_filter1d
import torch.nn.functional as F

# based on:
# https://github.com/TRAILab/CaDDN/blob/master/pcdet/models/backbones_3d/ffe/ddn_loss/ddn_loss.py


class DDNLoss(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2.0,
                 fg_weight=13,
                 bg_weight=1,
                 downsample_factor=1):
        """
        Initializes DDNLoss module
        Args:
            weight [float]: Loss function weight
            alpha [float]: Alpha value for Focal Loss
            gamma [float]: Gamma value for Focal Loss
            disc_cfg [dict]: Depth discretiziation configuration
            fg_weight [float]: Foreground loss weight
            bg_weight [float]: Background loss weight
            downsample_factor [int]: Depth map downsample factor
        """
        super().__init__()
        self.device = torch.cuda.current_device()
        self.balancer = Balancer(
            downsample_factor=downsample_factor,
            fg_weight=fg_weight,
            bg_weight=bg_weight)

        # Set loss function
        self.alpha = alpha
        self.gamma = gamma
        
        self.sigma = 1.0
        
        self.loss_func = FocalLoss(alpha=self.alpha, gamma=self.gamma, reduction="none")

    def build_target_depth_from_3dcenter(self, depth_logits, gt_boxes2d, gt_center_depth, num_gt_per_img):
        B, _, H, W = depth_logits.shape
        depth_maps = torch.zeros((B, H, W), device=depth_logits.device, dtype=depth_logits.dtype)

        # Set box corners
        gt_boxes2d[:, :2] = torch.floor(gt_boxes2d[:, :2])
        gt_boxes2d[:, 2:] = torch.ceil(gt_boxes2d[:, 2:])
        gt_boxes2d = gt_boxes2d.long()

        # Set all values within each box to True
        gt_boxes2d = gt_boxes2d.split(num_gt_per_img, dim=0)
        gt_center_depth = gt_center_depth.split(num_gt_per_img, dim=0)
        B = len(gt_boxes2d)
        for b in range(B):
            center_depth_per_batch = gt_center_depth[b]
            center_depth_per_batch, sorted_idx = torch.sort(center_depth_per_batch, dim=0, descending=True)
            gt_boxes_per_batch = gt_boxes2d[b][sorted_idx]
            for n in range(gt_boxes_per_batch.shape[0]):
                u1, v1, u2, v2 = gt_boxes_per_batch[n]
                depth_maps[b, v1:v2, u1:u2] = center_depth_per_batch[n]

        return depth_maps

    def gaussian_kernel(self, size: int, sigma: float):
        """Function to create a 1D Gaussian kernel."""
        x = torch.arange(-size // 2 + 1., size // 2 + 1.)
        kernel = torch.exp(-(x**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel
    
    def build_weighted_depth_from_logits(self, depth_logits):
        depth_num_bins = 80
        depth_min = 1e-3
        depth_max = 60.0
        bin_size = 2 * (depth_max - depth_min) / (depth_num_bins * (1 + depth_num_bins))
        bin_indice = torch.linspace(0, depth_num_bins - 1, depth_num_bins)
        bin_value = (bin_indice + 0.5).pow(2) * bin_size / 2 - bin_size / 8 + depth_min
        bin_value = torch.cat([bin_value, torch.tensor([depth_max])], dim=0)
        self.depth_bin_values = nn.Parameter(bin_value, requires_grad=False).to(depth_logits.device)
        
        '''
        # Create Gaussian kernel
        kernel_size = int(2 * self.sigma + 1)  # Ensure the kernel size is odd
        gaussian_kernel = self.gaussian_kernel(kernel_size, self.sigma).to(depth_logits.device)
        
        # Expand the kernel to apply it across depth_num_bins dimension
        gaussian_kernel = gaussian_kernel.view(1, 1, -1, 1, 1)  # Shape: (1, 1, kernel_size, 1, 1)

        # Pad the depth_logits tensor to apply the convolution
        padding = (kernel_size // 2, kernel_size // 2)
        padded_logits = F.pad(depth_logits.unsqueeze(2), (0, 0, 0, 0, padding[0], padding[1]), mode='reflect')
        
        # Apply the Gaussian filter along the depth_num_bins dimension
        smoothed_logits = F.conv3d(padded_logits, gaussian_kernel, groups=depth_logits.size(0)).squeeze(2)
        '''
        
        # Calculate the median value along the depth_num_bins dimension
        median_values = torch.median(depth_logits, dim=1, keepdim=True).values
        # Apply the threshold: set values below the median to zero
        thresholded_logits = torch.where(depth_logits >= median_values, depth_logits, torch.tensor(0.0).to(depth_logits.device))
        
        
        depth_probs = F.softmax(thresholded_logits, dim=1)
        weighted_depth = (depth_probs * self.depth_bin_values.reshape(1, -1, 1, 1)).sum(dim=1)

        return weighted_depth
    
    
    def bin_depths(self, depth_map, mode="LID", depth_min=1e-3, depth_max=60, num_bins=80, target=False):
        """
        Converts depth map into bin indices
        Args:
            depth_map [torch.Tensor(H, W)]: Depth Map
            mode [string]: Discretiziation mode (See https://arxiv.org/pdf/2005.13423.pdf for more details)
                UD: Uniform discretiziation
                LID: Linear increasing discretiziation
                SID: Spacing increasing discretiziation
            depth_min [float]: Minimum depth value
            depth_max [float]: Maximum depth value
            num_bins [int]: Number of depth bins
            target [bool]: Whether the depth bins indices will be used for a target tensor in loss comparison
        Returns:
            indices [torch.Tensor(H, W)]: Depth bin indices
        """
        if mode == "UD":
            bin_size = (depth_max - depth_min) / num_bins
            indices = ((depth_map - depth_min) / bin_size)
        elif mode == "LID":
            bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
            indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (depth_map - depth_min) / bin_size)
        elif mode == "SID":
            indices = num_bins * (torch.log(1 + depth_map) - math.log(1 + depth_min)) / \
                      (math.log(1 + depth_max) - math.log(1 + depth_min))
        else:
            raise NotImplementedError

        if target:
            # Remove indicies outside of bounds
            mask = (indices < 0) | (indices > num_bins) | (~torch.isfinite(indices))
            indices[mask] = num_bins

            # Convert to integer
            indices = indices.type(torch.int64)
       
        # Apply Gaussian noise to the depth indices
        #noise = torch.normal(mean=0, std=2, size=indices.shape, device=indices.device)
        #indices = indices +  noise.round().long()

        # Clamp indices to be within the valid range
        #indices = indices.clamp(0, num_bins)
        
        return indices

    def forward(self, depth_logits, gt_boxes2d, num_gt_per_img, gt_center_depth):
        """
        Gets depth_map loss
        Args:
            depth_logits: torch.Tensor(B, D+1, H, W)]: Predicted depth logits
            gt_boxes2d [torch.Tensor (B, N, 4)]: 2D box labels for foreground/background balancing
            num_gt_per_img:
            gt_center_depth:
        Returns:
            loss [torch.Tensor(1)]: Depth classification network loss
        """

        # Bin depth map to create target
        depth_maps = self.build_target_depth_from_3dcenter(depth_logits, gt_boxes2d, gt_center_depth, num_gt_per_img)
        depth_target = self.bin_depths(depth_maps, target=True)

        # Compute reg loss
        #weighted_depth = self.build_weighted_depth_from_logits(depth_logits)
        #reg_loss = F.smooth_l1_loss(weighted_depth, depth_maps, reduction='none')
        #reg_loss = self.balancer(loss=reg_loss, gt_boxes2d=gt_boxes2d, num_gt_per_img=num_gt_per_img)
        
        # Compute cls loss
        cls_loss = self.loss_func(depth_logits, depth_target)
        # Compute foreground/background balancing
        cls_loss = self.balancer(loss=cls_loss, gt_boxes2d=gt_boxes2d, num_gt_per_img=num_gt_per_img)
        
        loss = cls_loss

        return loss
