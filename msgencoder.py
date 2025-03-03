import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

def first_layer_film_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)

def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

class CustomMappingNetwork(nn.Module):
    def __init__(self, z_dim, map_hidden_dim, map_output_dim):
        super().__init__()



        self.network = nn.Sequential(nn.Linear(z_dim, map_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_hidden_dim),
                                    nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_hidden_dim),
                                    nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_output_dim))

        self.network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.network[-1].weight *= 0.25

    def forward(self, z):
        frequencies_offsets = self.network(z)
        frequencies = frequencies_offsets[..., :frequencies_offsets.shape[-1]//2]
        phase_shifts = frequencies_offsets[..., frequencies_offsets.shape[-1]//2:]

        return frequencies, phase_shifts


def frequency_init(freq):
    def init(m):
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)
    return init

class FiLMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, freq, phase_shift):
        x = self.layer(x)
        freq = freq.unsqueeze(1).expand_as(x)
        phase_shift = phase_shift.unsqueeze(1).expand_as(x)
        return torch.sin(freq * x + phase_shift)


# class TALLSIREN(nn.Module):
#     """Primary SIREN  architecture used in pi-GAN generators."""

#     def __init__(self, input_dim=2, z_dim=100, hidden_dim=256, output_dim=1, device=None):
#         super().__init__()
#         self.device = device
#         self.input_dim = input_dim
#         self.z_dim = z_dim
#         self.hidden_dim = hidden_dim
#         self.output_dim = output_dim

#         self.network = nn.ModuleList([
#             FiLMLayer(input_dim, hidden_dim),
#             FiLMLayer(hidden_dim, hidden_dim),
#             FiLMLayer(hidden_dim, hidden_dim),
#             FiLMLayer(hidden_dim, hidden_dim),
#             FiLMLayer(hidden_dim, hidden_dim),
#             FiLMLayer(hidden_dim, hidden_dim),
#             FiLMLayer(hidden_dim, hidden_dim),
#             FiLMLayer(hidden_dim, hidden_dim),
#         ])
#         self.final_layer = nn.Linear(hidden_dim, 1)

#         self.color_layer_sine = FiLMLayer(hidden_dim + 3, hidden_dim)
#         self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3), nn.Sigmoid())

#         self.mapping_network = CustomMappingNetwork(z_dim, 256, (len(self.network) + 1)*hidden_dim*2)

#         self.network.apply(frequency_init(25))
#         self.final_layer.apply(frequency_init(25))
#         self.color_layer_sine.apply(frequency_init(25))
#         self.color_layer_linear.apply(frequency_init(25))
#         self.network[0].apply(first_layer_film_sine_init)

#     def forward(self, input, z, ray_directions, **kwargs):
#         frequencies, phase_shifts = self.mapping_network(z)
#         return self.forward_with_frequencies_phase_shifts(input, frequencies, phase_shifts, ray_directions, **kwargs)

#     def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts, ray_directions, **kwargs):
#         frequencies = frequencies*15 + 30

#         x = input

#         for index, layer in enumerate(self.network):
#             start = index * self.hidden_dim
#             end = (index+1) * self.hidden_dim
#             x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])

#         sigma = self.final_layer(x)
#         rbg = self.color_layer_sine(torch.cat([ray_directions, x], dim=-1), frequencies[..., -self.hidden_dim:], phase_shifts[..., -self.hidden_dim:])
#         rbg = self.color_layer_linear(rbg)

#         return torch.cat([rbg, sigma], dim=-1)

class msg_FiLM(nn.Module):
    def __init__(self, message_dim, feature_dim, coordinate_dim, hidden_dim=256):
        super(msg_FiLM, self).__init__()
        self.message_dim = message_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        self.network = nn.ModuleList([
            FiLMLayer(feature_dim + coordinate_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])
        self.final_layer = nn.Linear(hidden_dim, feature_dim)

        self.mapping_network = CustomMappingNetwork(message_dim, 256, (len(self.network))*hidden_dim*2)

        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)

        
    def forward(self, message, voxel_vertex, voxel_embedds):
        frequencies, phase_shifts = self.mapping_network(message)
        return self.forward_with_frequencies_phase_shifts(voxel_vertex, voxel_embedds, frequencies, phase_shifts)
    
    def forward_with_frequencies_phase_shifts(self, voxel_vertex, voxel_embedds, frequencies, phase_shifts):
        frequencies = frequencies*15 + 30

        x = torch.cat([voxel_vertex, voxel_embedds], dim=-1)
        x = rearrange(x, 'b n l c -> b n (l c)')

        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index+1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])

        x = self.final_layer(x)
        x = rearrange(x, 'b n (l c) -> b n l c', l = 16)
        return x + voxel_embedds

