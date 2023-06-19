import torch
import torch.nn as nn


class NormScalingHead(nn.Module):
    def __init__(self, input_dim, scale_factor=1.0):
        super(NormScalingHead, self).__init__()
        self.input_dim = input_dim
        self.scale_factor = scale_factor

        # Create a learnable scaling factor
        self.scaling_factor = nn.Parameter(torch.ones(1, input_dim))

    def forward(self, x):
        # Normalize the input along the input_dim dimension
        norm_x = torch.norm(x, dim=self.input_dim, keepdim=True)

        # Scale the normalized input by the scaling factor
        scaled_x = self.scale_factor * self.scaling_factor * x / norm_x

        return scaled_x
