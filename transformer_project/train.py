import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle

# 📦 On importe les tenseurs déjà préparés depuis prepare_data.py
from prepare_data import X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

# 📦 On importe les modules personnalisés du projet
from modules.encoder import TransformerEncoder
from modules.classifier import TransformerClassifier

# --------------------------
# 🎯 HYPERPARAMÈTRES
# --------------------------
d_model = X_train_tensor.shape[1]  # Ici : 6 features
num_heads = 2                      # (non utilisé avec Mean-Field, conservé pour structure)
d_ff = 12
num_layers = 2
num_classes = 2
learning_rate = 0.001
num_epochs = 10
batch_size = 64

activation_functions = ["relu", "tanh", "sigmoid", "leakyrelu", "gelu"]
results = {}

# --------------------------
# 🔁 BOUCLE SUR LES ACTIVATIONS
# --------------------------
for activation in activation_functions:
    print(f"\n🔍 Test avec activation : {activation.upper()}")

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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    loss_list = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        loss_list.append(avg_loss)
        print(f"  Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, preds = torch.max(outputs, 1)
        accuracy = (preds == y_test_tensor).float().mean().item()
        print(f"  ✅ Accuracy sur test set : {accuracy:.4f}")

    results[activation] = {
        "loss": loss_list,
        "accuracy": accuracy
    }

    torch.save(model.state_dict(), f"model_{activation}.pth")

# --------------------------
# 💾 Sauvegarde des résultats
# --------------------------
with open("results_activation.pkl", "wb") as f:
    pickle.dump(results, f)

print("\n✅ Résultats sauvegardés dans results_activation.pkl")
