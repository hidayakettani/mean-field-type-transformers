import torch
import pickle
import pandas as pd
from modules.encoder import TransformerEncoder
from modules.classifier import TransformerClassifier

# === PARAMÈTRES DU MODÈLE (doivent correspondre à train.py) ===
d_model = 6
num_heads = 2
d_ff = 12
num_layers = 2
num_classes = 2

# === CHARGEMENT DES CSV ORIGINAUX POUR NOM → ID ===
drivers_df = pd.read_csv('data/drivers.csv')
constructors_df = pd.read_csv('data/constructors.csv')
races_df = pd.read_csv('data/races.csv')

# Dictionnaires nom→ID
driver_name_to_id = dict(zip(drivers_df['surname'].str.lower(), drivers_df['driverId']))
constructor_name_to_id = dict(zip(constructors_df['name'].str.lower(), constructors_df['constructorId']))
race_name_to_id = dict(zip(races_df['name'].str.lower(), races_df['raceId']))

# === CHARGEMENT DES ENCODEURS ===
with open('driver_encoder.pkl', 'rb') as f:
    driver_encoder = pickle.load(f)
with open('constructor_encoder.pkl', 'rb') as f:
    constructor_encoder = pickle.load(f)
with open('race_encoder.pkl', 'rb') as f:
    race_encoder = pickle.load(f)

encoders = {
    'driverId': driver_encoder,
    'constructorId': constructor_encoder,
    'raceId': race_encoder,
}

# === CHOIX INTERACTIF DE L’ACTIVATION ===
activation = input(" Quelle fonction d’activation ? (relu, tanh, sigmoid, leakyrelu, gelu): ").lower()
model_path = f"model_{activation}.pth"

# === INITIALISATION DU MODÈLE AVEC L’ACTIVATION CHOISIE ===
encoder = TransformerEncoder(
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    num_layers=num_layers,
    activation=activation
)
model = TransformerClassifier(encoder=encoder, d_model=d_model, num_classes=num_classes)
model.load_state_dict(torch.load(model_path))
model.eval()

# === SAISIE UTILISATEUR ===
print("\n=== Prédiction Formula 1 ===\n")
driver = input("Nom du pilote (ex: Hamilton) : ").lower()
constructor = input("Nom du constructeur (ex: Mercedes) : ").lower()
race = input("Nom du Grand Prix (ex: Monaco Grand Prix) : ").lower()
grid = int(input("Position sur la grille : "))
laps = int(input("Nombre de tours : "))
points = float(input("Nombre de points : "))

# === TRANSFORMATION EN IDs + ENCODAGE ===
try:
    driver_id = encoders['driverId'].transform([str(driver_name_to_id[driver])])[0]
    constructor_id = encoders['constructorId'].transform([str(constructor_name_to_id[constructor])])[0]
    race_id = encoders['raceId'].transform([str(race_name_to_id[race])])[0]
except KeyError as e:
    print(f"\n Erreur : le nom saisi est introuvable → {e}")
    exit()

# === CRÉATION DU TENSEUR D’ENTRÉE ===
features = [driver_id, constructor_id, race_id, grid, laps, points]
x = torch.tensor([features], dtype=torch.float32)

# === PRÉDICTION ===
with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=1).numpy()[0]
    prediction = probs.argmax()
    confidence = probs[prediction] * 100
    error_rate = (1 - probs[prediction]) * 100

# === AFFICHAGE DU RÉSULTAT ===
print("\nRésultat de la prédiction :\n")
if prediction == 1:
    print(f"{driver.title()} a GAGNÉ la course ({confidence:.2f}% de confiance)")
    print(f" Risque d’erreur : {error_rate:.2f}%")
else:
    print(f" {driver.title()} N’A PAS gagné la course ({confidence:.2f}% de confiance)")
    print(f"  Risque d’erreur : {error_rate:.2f}%")
