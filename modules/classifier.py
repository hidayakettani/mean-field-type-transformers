import torch
import torch.nn as nn

class TransformerClassifier(nn.Module):
    def __init__(self, encoder, d_model, num_classes):
        super(TransformerClassifier, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        # Comme x est [batch_size, features] ici, pas de token CLS, on garde tout
        # Si x a encore 3 dimensions, adapte, sinon :
        if x.dim() == 3:
            x = x[:, 0, :]  # prend le token CLS si présent
        else:
            x = x.view(x.size(0), -1)  # aplatisse au cas où
        return self.classifier(x)
