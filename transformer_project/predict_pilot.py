import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.encoder import TransformerEncoder
from modules.classifier import TransformerClassifier

# === PARAMÈTRES DU MODÈLE === #
d_model = 6
num_heads = 2
d_ff = 12
num_layers = 2
num_classes = 2

# === 💡 CHOIX FONCTION D'ACTIVATION === #
activation = input("👉 Quelle fonction d’activation ? (relu, tanh, sigmoid, leakyrelu, gelu): ").lower()

# === 🔢 SAISIE DES FEATURES DU PILOTE === #
print("🔧 Entrez les 6 valeurs du pilote (driverId, constructorId, raceId, grid, laps, points)")
try:
    values = [float(input(f"  Valeur {i+1} : ")) for i in range(6)]
except:
    print("⛔ Erreur de saisie. Assurez-vous d’entrer 6 nombres.")
    exit()

x_f1 = torch.tensor([[values]], dtype=torch.float32)  # Shape (1, 1, 6)

# === CRÉATION DU MODÈLE === #
encoder = TransformerEncoder(
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    num_layers=num_layers,
    activation=activation
)
model = TransformerClassifier(
    encoder=encoder,
    d_model=d_model,
    num_classes=num_classes
)

# === CHARGEMENT DES POIDS === #
path = f"model_{activation}.pth"
if not os.path.exists(path):
    print(f"⛔ Modèle non trouvé : {path}")
    exit()

model.load_state_dict(torch.load(path))
model.eval()

# === PRÉDICTION === #
with torch.no_grad():
    output = model(x_f1)
    probas = torch.softmax(output, dim=1)
    pred_class = torch.argmax(probas, dim=1).item()

# === 🖨️ AFFICHAGE DU RÉSULTAT === #
print("\n🔍 Résultat de la prédiction :")
print("Activation :", activation.upper())
print("Entrée     :", values)
print("Probabilités :", [round(p.item(), 4) for p in probas.squeeze(0)])
print("✅ Classe prédite :", pred_class)
