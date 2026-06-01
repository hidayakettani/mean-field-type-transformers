import torch
import torch.nn.functional as F

def mean_field_attention(Q, K, V, activation=torch.relu):
    """
    Implémente le mécanisme d’attention de type champ moyen
    Q, K, V : tensors de forme (batch, seq_len, d_model)
    activation : ReLU ou autre fonction positive
    """
    phi_Q = activation(Q)
    phi_K = activation(K)

    # Moyenne pondérée des valeurs
    context = torch.matmul(phi_K.transpose(-2, -1), V) / K.shape[1]

    # Produit avec φ(Q)
    output = torch.matmul(phi_Q, context)
    return output
