import matplotlib.pyplot as plt
import pickle
import os

# 📥 Charger les résultats sauvegardés depuis train.py
with open("results_activation.pkl", "rb") as f:
    results = pickle.load(f)

# 📁 Créer un dossier plots/ si non existant
os.makedirs("plots", exist_ok=True)

# -------------------------------
# 🔹 Courbes individuelles par fonction
# -------------------------------
for act in results:
    loss = results[act]["loss"]
    acc = results[act]["accuracy"]

    # Courbe de perte (loss)
    plt.figure()
    plt.plot(loss, marker='o', label=f"Loss - {act.upper()}")
    plt.title(f"Courbe de perte ({act.upper()})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/loss_{act}.png")
    plt.savefig(f"plots/loss_{act}.pdf")
    plt.close()

    # Courbe d'accuracy (point final uniquement ici)
    plt.figure()
    plt.bar([act.upper()], [acc], color='skyblue')
    plt.ylim(0, 1)
    plt.title(f"Accuracy finale ({act.upper()})")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(f"plots/acc_{act}.png")
    plt.savefig(f"plots/acc_{act}.pdf")
    plt.close()

# -------------------------------
# 🔹 Courbes combinées (toutes les activations sur 1 plot)
# -------------------------------
# Loss comparée
plt.figure()
for act in results:
    plt.plot(results[act]["loss"], label=act.upper())
plt.title("Comparaison des fonctions d’activation (Loss)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("plots/loss_all.png")
plt.savefig("plots/loss_all.pdf")
plt.close()

# Accuracy comparée
plt.figure()
acts = [a.upper() for a in results]
accs = [results[a]["accuracy"] for a in results]
plt.bar(acts, accs, color='lightgreen')
plt.title("Accuracy finale par activation")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("plots/acc_all.png")
plt.savefig("plots/acc_all.pdf")
plt.close()

print("✅ Toutes les courbes sont générées dans le dossier 'plots/'")
