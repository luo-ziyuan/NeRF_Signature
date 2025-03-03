# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import torch
import torch.nn as nn
from torchvision import transforms

### Load HiDDeN models
normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize (x - mean) / std
unnormalize_img = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]) # Unnormalize (x * std) + mean

class ConvBNRelu(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """
    def __init__(self, channels_in, channels_out):

        super(ConvBNRelu, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride=1, padding=1),
            nn.BatchNorm2d(channels_out, eps=1e-3, track_running_stats=False),
            # nn.LayerNorm([channels_out, 200, 200], eps=1e-5),
            nn.GELU()
        )

    def forward(self, x):
        # print(torch.any(torch.isnan(x) | torch.isinf(x)))
        # # x_np = x.detach().cpu().numpy()
        # print(x.mean())
        return self.layers(x)

    
class ConvBNRelu_original(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """
    def __init__(self, channels_in, channels_out):

        super(ConvBNRelu_original, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride=1, padding=1),
            nn.BatchNorm2d(channels_out, eps=1e-3),
            nn.GELU()
        )

    def forward(self, x):
        return self.layers(x)
# class HiddenDecoder(nn.Module):
#     """
#     Decoder module. Receives a watermarked image and extracts the watermark.
#     """
#     def __init__(self, num_blocks, num_bits, channels, redundancy=1):

#         super(HiddenDecoder, self).__init__()

#         layers = [ConvBNRelu(3, channels)]
#         for _ in range(num_blocks - 1):
#             layers.append(ConvBNRelu(channels, channels))

#         layers.append(ConvBNRelu(channels, num_bits*redundancy))
#         layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
#         self.layers = nn.Sequential(*layers)

#         self.linear = nn.Linear(num_bits*redundancy, num_bits*redundancy)

#         self.num_bits = num_bits
#         self.redundancy = redundancy

        

class HiddenDecoder(nn.Module):
    """
    Decoder module. Receives a watermarked image and extracts the watermark.
    The input image may have various kinds of noise applied to it,
    such as Crop, JpegCompression, and so on. See Noise layers for more.
    """
    def __init__(self, num_blocks, num_bits, channels):

        super(HiddenDecoder, self).__init__()

        layers = [ConvBNRelu_original(3, channels)]
        for _ in range(num_blocks - 1):
            layers.append(ConvBNRelu_original(channels, channels))

        layers.append(ConvBNRelu_original(channels, num_bits))
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)

        self.linear = nn.Linear(num_bits, num_bits)

    def forward(self, img_w):

        x = self.layers(img_w) # b d 1 1
        x = x.squeeze(-1).squeeze(-1) # b d
        x = self.linear(x) # b d
        return x
    
class HiddenDecoder_multi_views(nn.Module):
    """
    Decoder module. Receives a watermarked image and extracts the watermark.
    """
    def __init__(self, num_blocks, num_bits, input_ch, channels, redundancy=1):

        super(HiddenDecoder_multi_views, self).__init__()

        layers = [ConvBNRelu(input_ch, channels)]
        for _ in range(num_blocks - 1):
            layers.append(ConvBNRelu(channels, channels))

        layers.append(ConvBNRelu(channels, num_bits*redundancy))
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)

        self.linear = nn.Linear(num_bits*redundancy, num_bits*redundancy)

        self.num_bits = num_bits
        self.redundancy = redundancy
        
    def forward(self, img_w):
        # last_layer_params = self.layers.named_parameters()
        # for name, param in last_layer_params:
        #     print(name)
        #     print(torch.any(torch.isnan(param) | torch.isinf(param)))
        x = self.layers(img_w) # b d 1 1
        x = x.squeeze(-1).squeeze(-1) # b d
        x = self.linear(x)

        x = x.view(-1, self.num_bits, self.redundancy) # b k*r -> b k r
        x = torch.sum(x, dim=-1) # b k r -> b k

        return x

class HiddenEncoder(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self, num_blocks, num_bits, channels, last_tanh=True):
        super(HiddenEncoder, self).__init__()
        layers = [ConvBNRelu(3, channels)]

        for _ in range(num_blocks-1):
            layer = ConvBNRelu(channels, channels)
            layers.append(layer)

        self.conv_bns = nn.Sequential(*layers)
        self.after_concat_layer = ConvBNRelu(channels + 3 + num_bits, channels)

        self.final_layer = nn.Conv2d(channels, 3, kernel_size=1)

        self.last_tanh = last_tanh
        self.tanh = nn.Tanh()

    def forward(self, imgs, msgs):

        msgs = msgs.unsqueeze(-1).unsqueeze(-1) # b l 1 1
        msgs = msgs.expand(-1,-1, imgs.size(-2), imgs.size(-1)) # b l h w

        encoded_image = self.conv_bns(imgs)

        concat = torch.cat([msgs, encoded_image, imgs], dim=1)
        im_w = self.after_concat_layer(concat)
        im_w = self.final_layer(im_w)

        if self.last_tanh:
            im_w = self.tanh(im_w)

        return im_w

def get_hidden_decoder(num_bits, redundancy=1, num_blocks=7, channels=64):
    decoder = HiddenDecoder(num_blocks=num_blocks, num_bits=num_bits, channels=channels)
    return decoder

def get_hidden_decoder_multi_views(num_bits, redundancy=1, num_blocks=7, input_ch=3, channels=64):
    decoder = HiddenDecoder_multi_views(num_blocks=num_blocks, num_bits=num_bits, input_ch=input_ch, channels=channels, redundancy=redundancy)
    return decoder

def get_hidden_decoder_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # decoder_ckpt = { k.replace('module.', '').replace('decoder.', '') : v for k,v in ckpt['encoder_decoder'].items() if 'decoder' in k}
    decoder_ckpt = ckpt
    return decoder_ckpt

def get_hidden_encoder(num_bits, num_blocks=4, channels=64):
    encoder = HiddenEncoder(num_blocks=num_blocks, num_bits=num_bits, channels=channels)
    return encoder

def get_hidden_encoder_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    encoder_ckpt = { k.replace('module.', '').replace('encoder.', '') : v for k,v in ckpt['encoder_decoder'].items() if 'encoder' in k}
    return encoder_ckpt