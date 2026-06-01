import pandas as pd
import os

# 📁 Crée le dossier "data" dans transformer_project/ s'il n'existe pas
os.makedirs("transformer_project/data", exist_ok=True)

# 📦 Charge les fichiers CSV depuis le dossier data/
results = pd.read_csv("data/results.csv")
drivers = pd.read_csv("data/drivers.csv")
constructors = pd.read_csv("data/constructors.csv")
races = pd.read_csv("data/races.csv")

# 🔗 Fusionne les données pour créer un dataset complet
df = results.merge(drivers, on="driverId", how="left") \
            .merge(constructors, on="constructorId", how="left") \
            .merge(races, on="raceId", how="left")

# 🎯 Crée la colonne cible : 1 si le pilote a gagné, 0 sinon
df["target"] = (df["positionOrder"] == 1).astype(int)

# 🧹 Sélectionne les colonnes utiles
final_df = df[[
    "driverId", "constructorId", "raceId", "grid", "laps", "points", "target"
]]

# 💾 Sauvegarde dans le bon dossier
final_df.to_csv("transformer_project/data/f1_final_dataset.csv", index=False)

print("✅ Le fichier 'f1_final_dataset.csv' a été créé avec succès dans le dossier transformer_project/data/")
