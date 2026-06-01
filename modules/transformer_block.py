import torch
import torch.nn as nn
from modules.norm import Norm
from modules.mean_field_attention import mean_field_attention
from modules.feedforward import FeedForward  # On importe notre bloc FeedForward dynamique

class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_hidden, num_heads=1, activation="relu"):
        super().__init__()
        # Normalisation avant attention
        self.norm1 = Norm(d_model)
        # Normalisation avant FeedForward
        self.norm2 = Norm(d_model)
        # Mécanisme d’attention : ici mean-field fixé
        self.attention = mean_field_attention
        # FeedForward : on passe la fonction d’activation choisie
        self.ff = FeedForward(d_model, d_hidden, activation=activation)

    def forward(self, x):
        # 1. Normalisation avant l’attention
        norm_x = self.norm1(x)
        # 2. Application de l’attention (ici mean-field)
        attn_out = self.attention(norm_x, norm_x, norm_x)
        # 3. Résidu : ajout de la sortie attentionnelle
        x = x + attn_out
        # 4. Normalisation puis passage par FeedForward
        x = x + self.ff(self.norm2(x))
        return x
