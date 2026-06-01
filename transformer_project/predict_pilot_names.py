# Importation des bibliothèques nécessaires
import torch
import torch.nn as nn
import pickle
import pandas as pd

# Importation des modules personnalisés (définis dans le projet)
from modules.encoder import TransformerEncoder
from modules.classifier import TransformerClassifier

# Chargement des objets LabelEncoder utilisés à l'entraînement
with open("driver_encoder.pkl", "rb") as f:
    driver_encoder = pickle.load(f)
with open("constructor_encoder.pkl", "rb") as f:
    constructor_encoder = pickle.load(f)
with open("race_encoder.pkl", "rb") as f:
    race_encoder = pickle.load(f)

# Chargement des fichiers CSV contenant les données brutes (noms des pilotes, écuries, courses)
drivers_df = pd.read_csv("data/drivers.csv")
constructors_df = pd.read_csv("data/constructors.csv")
races_df = pd.read_csv("data/races.csv")

# Fonction pour retrouver l'ID d'un pilote à partir de son nom
def get_driver_id(name):
    row = drivers_df[drivers_df['surname'].str.lower() == name.lower()]
    if row.empty:
        raise ValueError(f"Pilote '{name}' introuvable.")
    return str(row.iloc[0]['driverId'])

# Fonction pour retrouver l'ID d'une écurie à partir de son nom
def get_constructor_id(name):
    row = constructors_df[constructors_df['name'].str.lower() == name.lower()]
    if row.empty:
        raise ValueError(f"Écurie '{name}' introuvable.")
    return str(row.iloc[0]['constructorId'])

# Fonction pour retrouver l'ID d'un Grand Prix à partir de son nom
def get_race_id(name):
    row = races_df[races_df['name'].str.lower() == name.lower()]
    if row.empty:
        raise ValueError(f"Grand Prix '{name}' introuvable.")
    return str(row.iloc[0]['raceId'])

# Saisie de la fonction d'activation à utiliser
activation = input("Quelle fonction d’activation ? (relu, tanh, sigmoid, leakyrelu, gelu): ").lower()
model_path = f"model_{activation}.pth"

# Saisie des informations utilisateur (noms lisibles)
pilot_name = input("Nom du pilote (ex: Hamilton): ")
team_name = input("Nom de l’écurie (ex: Mercedes): ")
race_name = input("Nom du Grand Prix (ex: Monaco Grand Prix): ")
grid = int(input("Position sur la grille : "))
laps = int(input("Nombre de tours : "))
points = float(input("Nombre de points : "))

# Conversion des noms en IDs numériques
try:
    driver_id = get_driver_id(pilot_name)
    constructor_id = get_constructor_id(team_name)
    race_id = get_race_id(race_name)
except ValueError as e:
    print("\nErreur :", e)
    exit()

# Encodage des IDs avec les encodeurs entraînés
driver_encoded = driver_encoder.transform([driver_id])[0]
constructor_encoded = constructor_encoder.transform([constructor_id])[0]
race_encoded = race_encoder.transform([race_id])[0]

# Création du vecteur d'entrée au format PyTorch
input_tensor = torch.tensor([[driver_encoded, constructor_encoded, race_encoded, grid, laps, points]], dtype=torch.float32)

# Chargement du modèle Transformer avec la fonction d’activation choisie
d_model = 6  # Nombre de caractéristiques en entrée
encoder = TransformerEncoder(d_model=d_model, num_heads=2, d_ff=12, num_layers=2, activation=activation)
model = TransformerClassifier(encoder=encoder, d_model=d_model, num_classes=2)

# Vérifie que le fichier de modèle existe
try:
    model.load_state_dict(torch.load(model_path))
except FileNotFoundError:
    print(f"Le fichier du modèle '{model_path}' est introuvable.")
    exit()

model.eval()  # Passage en mode évaluation (pas de dropout, etc.)

# Prédiction avec softmax
with torch.no_grad():
    output = model(input_tensor)  # Logits du modèle
    proba = torch.softmax(output, dim=1).numpy()[0]  # Probabilités de chaque classe
    predicted_class = int(torch.argmax(output, dim=1).item())  # Classe prédite (0 ou 1)
    win_proba = proba[1] * 100  # Pourcentage de chance de victoire
    lose_proba = proba[0] * 100  # Pourcentage de non-victoire

# Calcul de la CrossEntropyLoss pour cet exemple
# Cette valeur donne une idée de l’incertitude globale sur la prédiction
criterion = nn.CrossEntropyLoss()
fake_target = torch.tensor([predicted_class], dtype=torch.long)  # Cible simulée (même que prédiction)
loss_value = criterion(output, fake_target).item()  # Calcul de la perte (loss)

# Affichage clair du résultat
print("\n" + "=" * 50)
print(f"Résultat de prédiction pour {pilot_name} ({team_name})")
print(f"Course : {race_name}")
print("-" * 50)

if predicted_class == 1:
    print(f"PRÉDICTION : {pilot_name} a GAGNÉ la course.")
else:
    print(f"PRÉDICTION : {pilot_name} N’A PAS gagné la course.")

print(f"\nProbabilité de GAGNER : {win_proba:.2f}%")
print(f"Probabilité d'ERREUR : {lose_proba:.2f}%")
print(f"Valeur de la fonction de perte (loss) : {loss_value:.4f}")
print("=" * 50)
