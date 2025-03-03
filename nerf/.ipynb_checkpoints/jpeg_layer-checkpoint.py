# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional

import kornia.augmentation as K
from kornia.augmentation import AugmentationBase2D

from . import utils_img

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DiffJPEG(nn.Module):
    def __init__(self, quality=50):
        super().__init__()
        self.quality = quality
    
    def forward(self, x):
        with torch.no_grad():
            img_clip = utils_img.clamp_pixel(x)
            img_jpeg = utils_img.jpeg_compress(img_clip, self.quality)
            img_gap = img_jpeg - x
            img_gap = img_gap.detach()
        img_aug = x+img_gap
        return img_aug

class RandomDiffJPEG(AugmentationBase2D):
    def __init__(self, p, low=10, high=100) -> None:
        super().__init__(p=p)
        self.diff_jpegs = [DiffJPEG(quality=qf).to(device) for qf in range(low,high,10)]

    def generate_parameters(self, input_shape: torch.Size):
        qf = torch.randint(high=len(self.diff_jpegs), size=input_shape[0:1])
        return dict(qf=qf)

    def compute_transformation(self, input, params, flags):
        return self.identity_matrix(input)

    def apply_transform(self, input, params, *args, **kwargs):
        B, C, H, W = input.shape
        qf = params['qf']
        output = torch.zeros_like(input)
        for ii in range(B):
            output[ii] = self.diff_jpegs[qf[ii]](input[ii:ii+1])
        return output

class RandomBlur(AugmentationBase2D):
    def __init__(self, blur_size, p=1) -> None:
        super().__init__(p=p)
        self.gaussian_blurs = [K.RandomGaussianBlur(kernel_size=(kk,kk), sigma= (kk*0.15 + 0.35, kk*0.15 + 0.35)) for kk in range(1,int(blur_size),2)]

    def generate_parameters(self, input_shape: torch.Size):
        blur_strength = torch.randint(high=len(self.gaussian_blurs), size=input_shape[0:1])
        return dict(blur_strength=blur_strength)

    def compute_transformation(self, input, params, flags):
        return self.identity_matrix(input)

    def apply_transform(self, input, params, *args, **kwargs):
        B, C, H, W = input.shape
        blur_strength = params['blur_strength']
        output = torch.zeros_like(input)
        for ii in range(B):
            output[ii] = self.gaussian_blurs[blur_strength[ii]](input[ii:ii+1])
        return output

class JPEGLayer(nn.Module):
    def __init__(self, diff_jpeg=50):
        super(JPEGLayer, self).__init__()
        self.diff_jpeg = RandomDiffJPEG(p=1, low=diff_jpeg).to(device)
    
    def forward(self, input):
        input = self.diff_jpeg(input)
        return input
