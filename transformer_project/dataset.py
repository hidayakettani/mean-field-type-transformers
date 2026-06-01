import torch
from torch.utils.data import Dataset

class DummySequenceDataset(Dataset):
    def __init__(self, num_samples=100, seq_len=5, d_model=4):
        # Création aléatoire de séquences de vecteurs
        self.inputs = torch.randn(num_samples, seq_len, d_model)
        self.labels = torch.randint(0, 2, (num_samples,))  # classes 0 ou 1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]
