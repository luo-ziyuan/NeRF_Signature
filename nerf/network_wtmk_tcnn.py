import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import tinycudann as tcnn
from activation import trunc_exp
from .renderer_wtmk import NeRFRenderer
from einops import rearrange, repeat
from hash_encoding_wtmk_bit import HashEmbedder as HashEmbedder_msg
from hash_encoding import HashEmbedder
from .hidden_models import get_hidden_decoder, get_hidden_decoder_multi_views, get_hidden_decoder_ckpt, normalize_img

class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 bound=1,
                 message_dim=16,
                 n_views=1,
                 finetune_decoder=False,
                 **kwargs
                 ):
        super().__init__(bound, **kwargs)

        self.finetune_decoder = finetune_decoder
        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim

        per_level_scale = np.exp2(np.log2(2048 * bound / 16) / (16 - 1))

        self.per_level_scale = per_level_scale
        
        self.encoder = HashEmbedder(bounding_box=(0, 1), n_levels=16, n_features_per_level=2,\
            log2_hashmap_size=19, base_resolution=16, finest_resolution=2048)
        
        self.msg_encoder = HashEmbedder_msg(bounding_box=(0, 1), n_levels=message_dim*2, n_features_per_level=2,\
            log2_hashmap_size=19, base_resolution=2048, finest_resolution=2048, message_dim=message_dim)
        

        msg_decoder = get_hidden_decoder_multi_views(num_bits=1, redundancy=1,
                                                 num_blocks=8, input_ch=n_views*3, channels=64)
        self.msg_decoder = msg_decoder
        self.normalization = normalize_img
    
        self.sigma_net = tcnn.Network(
            n_input_dims=32,
            n_output_dims=1 + self.geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        )

        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color

        self.encoder_dir = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        self.in_dim_color = self.encoder_dir.n_output_dims + self.geo_feat_dim

        self.color_net = tcnn.Network(
            n_input_dims=self.in_dim_color,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim_color,
                "n_hidden_layers": num_layers_color - 1,
            },
        )
        
        if finetune_decoder:
            for param in [*self.encoder.parameters(), *self.color_net.parameters(), *self.sigma_net.parameters(), *self.msg_encoder.parameters()]:
                param.requires_grad = False
        else:
            for param in [*self.encoder.parameters(), *self.color_net.parameters(), *self.sigma_net.parameters()]:
                param.requires_grad = False

    def forward(self, x, d, message):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        # sigma
        x = (x + self.bound) / (2 * self.bound) # to [0, 1]
        x_feature = self.encoder(x)
        if message is not None:
            msg_feature = self.msg_encoder(x, message)
#             x_feature = x_feature + msg_feature
            x_feature[:, -2:] = x_feature[:, -2:] + msg_feature
        h = self.sigma_net(x_feature)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # color
        d = (d + 1) / 2 # tcnn SH encoding requires inputs to be in [0, 1]
        d = self.encoder_dir(d)

        #p = torch.zeros_like(geo_feat[..., :1]) # manual input padding
        h = torch.cat([d, geo_feat], dim=-1)
        h = self.color_net(h)
        
        # sigmoid activation for rgb
        color = torch.sigmoid(h)

        return sigma, color

    def density(self, x, message):
        # x: [N, 3], in [-bound, bound]

        x = (x + self.bound) / (2 * self.bound) # to [0, 1]
        x_feature = self.encoder(x)
        if message is not None:
            msg_feature = self.msg_encoder(x, message)
            x_feature[:, -2:] = x_feature[:, -2:] + msg_feature
        h = self.sigma_net(x_feature)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
        }

    # allow masked inference
    def color(self, x, d, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        x = (x + self.bound) / (2 * self.bound) # to [0, 1]

        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device) # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

        # color
        d = (d + 1) / 2 # tcnn SH encoding requires inputs to be in [0, 1]
        d = self.encoder_dir(d)

        h = torch.cat([d, geo_feat], dim=-1)
        h = self.color_net(h)
        
        # sigmoid activation for rgb
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype) # fp16 --> fp32
        else:
            rgbs = h

        return rgbs        

    # optimizer utils
    def get_params(self, lr):
        if self.finetune_decoder:
            params = [
                {'params': self.msg_decoder.parameters(), 'lr': lr},
            ]
        else:
            params = [
                {'params': self.msg_encoder.parameters(), 'lr': lr},
                {'params': self.msg_decoder.parameters(), 'lr': lr},
            ]
        
        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})
        
        return params