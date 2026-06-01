import torch
import torch.nn as nn  # On importe le module torch.nn pour créer les couches de neurones

# Fonction utilitaire pour choisir dynamiquement la fonction d’activation
def get_activation(name):
    name = name.lower()  # On convertit le nom en minuscule pour éviter les erreurs (ReLU vs relu)
    if name == "relu":
        return nn.ReLU()
    elif name == "leakyrelu":
        return nn.LeakyReLU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "gelu":
        return nn.GELU()
    else:
        # Si la fonction n’est pas reconnue, on lève une erreur explicite
        raise ValueError(f"Fonction d’activation inconnue : {name}")

# Définition de la classe FeedForward, qui hérite de nn.Module
class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden, activation="relu"):
        super(FeedForward, self).__init__()  # Appel au constructeur de nn.Module
        # Première couche linéaire : réduit ou agrandit la dimension
        self.linear1 = nn.Linear(d_model, d_hidden)
        # Fonction d’activation dynamique (ReLU, Tanh, etc.)
        self.activation = get_activation(activation)
        # Deuxième couche linéaire : ramène à la dimension initiale
        self.linear2 = nn.Linear(d_hidden, d_model)

    def forward(self, x):
        # Passage successif par les deux couches et la fonction d’activation
        # x -> linear1 -> activation -> linear2 -> output
        return self.linear2(self.activation(self.linear1(x)))
