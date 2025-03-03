import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import tinycudann as tcnn
from activation import trunc_exp
from .renderer_wtmk import NeRFRenderer
from einops import rearrange, repeat
from hash_encoding import HashEmbedder
from .hidden_models import get_hidden_decoder, get_hidden_decoder_multi_views, get_hidden_decoder_ckpt, normalize_img


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 encoding="HashGrid",
                 encoding_dir="SphericalHarmonics",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 bound=1,
                 message_dim=16,
                 msg_decoder_path=None,
                 msg_block_length=8,
                 msg_log_table=19,
                 n_views=1,
                 **kwargs
                 ):
        super().__init__(bound, **kwargs)

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim

        per_level_scale = np.exp2(np.log2(2048 * bound / 16) / (16 - 1))

        self.per_level_scale = per_level_scale

        self.block_length = msg_block_length

#         self.message_block = np.int(np.ceil(message_dim / msg_block_length))

#         self.encoder = tcnn.Encoding(
#             n_input_dims=3,
#             encoding_config={
#                 "otype": "HashGrid",
#                 "n_levels": 16,
#                 "n_features_per_level": 2,
#                 "log2_hashmap_size": 19,
#                 "base_resolution": 16,
#                 "per_level_scale": per_level_scale,
#             },
#         )
        self.encoder = HashEmbedder(bounding_box=(0, 1), n_levels=16, n_features_per_level=2,\
            log2_hashmap_size=19, base_resolution=16, finest_resolution=2048)

#         self.msg_encoder = HashEmbedder_msg(bounding_box=(0, 1), n_levels=message_dim*2, n_features_per_level=2,\
#             log2_hashmap_size=19, base_resolution=2048, finest_resolution=2048, message_dim=message_dim)
#         self.msg_encoder = Encoder_Tri_MLP_f(D=3, W=256, input_ch=60, input_ch_color=4, input_ch_message=message_dim,
#                                   input_ch_views=24, output_ch=3,
#                                   skips=[-1], use_viewdirs=True)

        msg_decoder = get_hidden_decoder(num_bits=message_dim, redundancy=1,
                                                 num_blocks=8, channels=64)
        linear = nn.Linear(message_dim, message_dim, bias=True)
        msg_decoder = nn.Sequential(msg_decoder, linear)
        if msg_decoder_path != "None":
            ckpt = get_hidden_decoder_ckpt(msg_decoder_path)
            msg_decoder.load_state_dict(ckpt)
            msg_decoder.eval()
        self.msg_decoder = msg_decoder
        self.normalization = normalize_img

#         self.msg_decoder_all = nn.ModuleList()
#         for _ in range(message_dim):
#             self.msg_decoder_all.append(get_hidden_decoder(num_bits=1, redundancy=1,
#                                                  num_blocks=8, channels=64))
#         self.normalization = normalize_img

#         msg_decoder = utils_model.get_hidden_decoder(num_bits=params.num_bits, redundancy=params.redundancy, num_blocks=params.decoder_depth, channels=params.decoder_channels).to(device)
#         ckpt = utils_model.get_hidden_decoder_ckpt(params.msg_decoder_path)
#         print(msg_decoder.load_state_dict(ckpt, strict=False))
#         msg_decoder.eval()
#         msg_decoder = torch.jit.load("msgdecoder/dec_48b_whit.pt").to(device)
    
#         self.msg_decoder = msg_decoder
#         self.normalization = normalize_img
    
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
        
#         self.encoder_pos = tcnn.Encoding(
#             n_input_dims=3,
#             encoding_config={
#                 "otype": "Frequency",
#                 "n_frequencies": 10,
#             },
#         )
        
#         self.encoder_dir_2 = tcnn.Encoding(
#             n_input_dims=3,
#             encoding_config={
#                 "otype": "Frequency",
#                 "n_frequencies": 4,
#             },
#         )

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
        
        
        for param in [*self.msg_decoder.parameters()]:
            param.requires_grad = False

#         for param in [*self.msg_decoder.parameters()]:
#             param.requires_grad = False
    def forward(self, x, d, message):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        # sigma
        x = (x + self.bound) / (2 * self.bound) # to [0, 1]
        x_feature = self.encoder(x)
#         if message is not None:
# #             message_block = []
# #             for i in range(self.message_block):
# #                 message_block.append(sum([message[j + self.block_length * i] * (2 ** j) for j in range(self.block_length)]))
# #             message_block = torch.stack(message_block, dim=-1).int()
#             msg_feature = self.msg_encoder(x, message)
# #             x_feature = x_feature + msg_feature
#             x_feature[:, -2:] = x_feature[:, -2:] + msg_feature
# #             x_feature[:, 0:2] = x_feature[:, 0:2] + msg_feature

#             # msg_feature = self.msg_encoder(x, message)
#             # x_feature[:, -2:] = x_feature[:, -2:] + msg_feature
        h = self.sigma_net(x_feature)

        #sigma = F.relu(h[..., 0])
        sigma_rough = h[..., 0]
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # color
        d_1 = (d + 1) / 2 # tcnn SH encoding requires inputs to be in [0, 1]
        d_1 = self.encoder_dir(d_1)

        #p = torch.zeros_like(geo_feat[..., :1]) # manual input padding
        h = torch.cat([d_1, geo_feat], dim=-1)
        h = self.color_net(h)
        
        # sigmoid activation for rgb
        color = torch.sigmoid(h)

#         if message is not None:
#             x = 2*x-1
#             x = self.encoder_pos(x)
#             d = self.encoder_dir_2(d)
            
#             color = self.msg_encoder(x, d, color, sigma_rough, message)
        return sigma, color

    def density(self, x, message):
        # x: [N, 3], in [-bound, bound]

        x = (x + self.bound) / (2 * self.bound) # to [0, 1]
        x_feature = self.encoder(x)
#         if message is not None:
# #             message_block = []
# #             for i in range(self.message_block):
# #                 message_block.append(sum([message[j + 8 * i] * (2 ** j) for j in range(8)]))
# #             message_block = torch.stack(message_block, dim=-1).int()
#             msg_feature = self.msg_encoder(x, message)
# #             x_feature = x_feature + msg_feature
#             x_feature[:, -2:] = x_feature[:, -2:] + msg_feature
# #             x_feature[:, 0:2] = x_feature[:, 0:2] + msg_feature

#             # msg_feature = self.msg_encoder(x, message)
#             # x_feature[:, -2:] = x_feature[:, -2:] + msg_feature
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

        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr},
        ]
        # for i in range(len(self.msg_decoder_all)):
        #     params.append({'params': self.msg_decoder_all[i].parameters(), 'lr': lr})
        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})
        
        return params
