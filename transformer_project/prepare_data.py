import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import pickle  # <-- ajout pour sauvegarder les encodeurs

# ✅ Chargement du dataset fusionné (chemin corrigé)
df = pd.read_csv("transformer_project/data/f1_final_dataset.csv")

# Colonnes à encoder (catégorielles)
categorical_cols = ['driverId', 'constructorId', 'raceId']

# Dictionnaire pour garder les encodeurs
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le  # On sauvegarde chaque encodeur dans le dictionnaire

# Sélectionne les features et la cible
X = df[['driverId', 'constructorId', 'raceId', 'grid', 'laps', 'points']].values
y = df['target'].values

# Divise en train/test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convertis en tenseurs PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

print("✅ Données préparées :")
print("Train X shape :", X_train_tensor.shape)
print("Train y shape :", y_train_tensor.shape)
print("Test X shape  :", X_test_tensor.shape)
print("Test y shape  :", y_test_tensor.shape)

# ✅ Sauvegarde des encodeurs
with open('driver_encoder.pkl', 'wb') as f:
    pickle.dump(encoders['driverId'], f)
with open('constructor_encoder.pkl', 'wb') as f:
    pickle.dump(encoders['constructorId'], f)
with open('race_encoder.pkl', 'wb') as f:
    pickle.dump(encoders['raceId'], f)

print("✅ Encodeurs sauvegardés dans des fichiers .pkl")
