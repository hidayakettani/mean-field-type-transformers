# predict_wrapper.py
import os
import glob
import torch
import pickle
import pandas as pd
from modules.encoder import TransformerEncoder
from modules.classifier import TransformerClassifier

# === PARAMÈTRES DU MODÈLE (doivent correspondre au training) ===
d_model = 6
num_heads = 2
d_ff = 12
num_layers = 2
num_classes = 2

# === CHARGEMENT DES CSV POUR NOM -> ID ===
drivers_df = pd.read_csv('data/drivers.csv')
constructors_df = pd.read_csv('data/constructors.csv')
races_df = pd.read_csv('data/races.csv')

driver_name_to_id = dict(zip(drivers_df['surname'].str.lower(), drivers_df['driverId']))
constructor_name_to_id = dict(zip(constructors_df['name'].str.lower(), constructors_df['constructorId']))
race_name_to_id = dict(zip(races_df['name'].str.lower(), races_df['raceId']))

# === CHARGEMENT DES ENCODEURS PICKLE ===
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

# === DÉTECTION DU FICHIER DE MODÈLE ===
# On tente de trouver un model_*.pth automatiquement
model_files = glob.glob("model_*.pth")
if not model_files:
    raise FileNotFoundError("Aucun fichier model_*.pth trouvé dans le dossier. Place ton model_{activation}.pth ici.")
# Si plusieurs, on prend le premier (ou tu peux spécifier)
model_path = model_files[0]
# déduire l'activation depuis le nom (model_relu.pth -> 'relu')
activation = os.path.basename(model_path).replace("model_", "").replace(".pth", "")

# === INSTANTIATION DU MODÈLE ===
encoder = TransformerEncoder(
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    num_layers=num_layers,
    activation=activation
)
model = TransformerClassifier(encoder=encoder, d_model=d_model, num_classes=num_classes)

# charger sur CPU pour éviter les erreurs CUDA sur machine sans GPU
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

def predict_from_dict(data: dict) -> dict:
    """
    Entrée attendue:
    {
      "driver": "hamilton",
      "constructor": "mercedes",
      "race": "monaco grand prix",
      "grid": 1,
      "laps": 78,
      "points": 25.0
    }
    """
    # validation minimale
    required = ["driver", "constructor", "race", "grid", "laps", "points"]
    for k in required:
        if k not in data:
            return {"error": f"Missing field: {k}"}

    try:
        driver_key = data["driver"].lower()
        constructor_key = data["constructor"].lower()
        race_key = data["race"].lower()

        driver_id = encoders['driverId'].transform([str(driver_name_to_id[driver_key])])[0]
        constructor_id = encoders['constructorId'].transform([str(constructor_name_to_id[constructor_key])])[0]
        race_id = encoders['raceId'].transform([str(race_name_to_id[race_key])])[0]
    except KeyError as e:
        return {"error": f"Nom introuvable: {e}. Vérifie l'orthographe et utilise les noms tels qu'ils apparaissent dans data/*.csv (minuscules)."}

    try:
        features = [int(driver_id), int(constructor_id), int(race_id), int(data["grid"]), int(data["laps"]), float(data["points"])]
        x = torch.tensor([features], dtype=torch.float32)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            prediction = int(probs.argmax())
            confidence = float(probs[prediction]) * 100
            error_rate = (1 - probs[prediction]) * 100

        return {
            "driver": data["driver"].title(),
            "prediction": "WON" if prediction == 1 else "DID NOT WIN",
            "confidence": round(confidence, 2),
            "error_rate": round(error_rate, 2),
            "model_used": os.path.basename(model_path),
            "activation": activation
        }
    except Exception as e:
        return {"error": f"Runtime error: {e}"}
