import torch
import torch.nn as nn

class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(Norm, self).__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x):
        return self.norm(x)
