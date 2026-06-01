import torch
import torch.nn as nn
from modules.transformer_block import TransformerBlock  # On importe le bloc Transformer de base

# Définition de la classe TransformerEncoder
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, activation="relu"):
        super().__init__()  # Initialisation du module parent (nn.Module)

        # On crée une liste de blocs Transformer (chaque bloc = TransformerBlock)
        # Chaque bloc reçoit les paramètres suivants :
        # - d_model : dimension des vecteurs d’entrée
        # - num_heads : nombre de têtes d’attention (pas utilisé ici car mean-field)
        # - d_hidden (appelé d_ff ici) : taille de la couche cachée dans le FeedForward
        # - activation : la fonction d’activation utilisée dans le bloc FeedForward (ReLU, Tanh, etc.)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_hidden=d_ff,
                activation=activation
            )
            for _ in range(num_layers)  # On répète cette structure autant de fois que le nombre de couches
        ])

    def forward(self, x):
        # On applique chaque couche TransformerBlock l'une après l'autre
        for layer in self.layers:
            x = layer(x)  # Passage à travers un bloc Transformer
        return x  # On retourne le résultat final après tous les blocs
