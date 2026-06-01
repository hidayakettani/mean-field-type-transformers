from norm import Norm
import torch
from modules.norm import Norm
from modules.feedforward import FeedForward
from modules.mean_field_attention import mean_field_attention
from modules.transformer_block import TransformerBlock
from modules.encoder import TransformerEncoder
from modules.positional_encoding import PositionalEncoding
from modules.classifier import TransformerClassifier



# Test LayerNorm
norm = Norm(d_model=4)
x = torch.randn(2, 5, 4)  # batch=2, seq_len=5, d_model=4
output = norm(x)
print("Résultat normalisé :")
print(output)
from mean_field_attention import mean_field_attention

# Tenseurs simulés Q=K=V (self-attention)
qkv = torch.randn(2, 5, 4)
att_output = mean_field_attention(qkv, qkv, qkv)
print("Résultat de l’attention champ moyen :")
print(att_output)
from feedforward import FeedForward

ffn = FeedForward(d_model=4, d_hidden=8)
ff_output = ffn(torch.randn(2, 5, 4))
print("Résultat FeedForward :")
print(ff_output)
from transformer_block import TransformerBlock

block = TransformerBlock(d_model=4, d_hidden=8)
x_input = torch.randn(2, 5, 4)
output = block(x_input)
print("Résultat du bloc Transformer complet :")
print(output)
from encoder import TransformerEncoder
encoder = TransformerEncoder(d_model=4, d_hidden=8, num_layers=3)
x_input = torch.randn(2, 5, 4)
encoded_output = encoder(x_input)
print("Résultat de l’encodeur empilé :")
print(encoded_output)
from positional_encoding import PositionalEncoding

pos_enc = PositionalEncoding(d_model=4)
x_input = torch.zeros(1, 10, 4)  # batch=1, seq_len=10, d_model=4
x_pos = pos_enc(x_input)

print("Encodage positionnel :")
print(x_pos)
from classifier import TransformerClassifier

encoder = TransformerEncoder(d_model=4, d_hidden=8, num_layers=3)
model = TransformerClassifier(encoder, d_model=4, num_classes=2)

x_input = torch.randn(2, 5, 4)
output = model(x_input)
print("Sortie finale du modèle (classification) :")
print(output)

