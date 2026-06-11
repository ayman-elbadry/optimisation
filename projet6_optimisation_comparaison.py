import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# -------------------------------------------------------------------------
# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES (MNIST 3 vs 8)
# -------------------------------------------------------------------------
print("1. Téléchargement et filtrage du dataset MNIST (chiffres 3 vs 8)...")
# fetch_openml télécharge le dataset MNIST officiel
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X_all, y_all = mnist.data, mnist.target.astype(int)

# Filtrage pour ne garder que les classes 3 et 8
mask = (y_all == 3) | (y_all == 8)
X_filtered = X_all[mask]
y_filtered = y_all[mask]

# Conversion des labels : 3 -> 0 et 8 -> 1 (pour la classification binaire)
y_filtered = np.where(y_filtered == 3, 0, y_filtered)
y_filtered = np.where(y_filtered == 8, 1, y_filtered)

# Échantillonnage pour accélérer l'entraînement (5000 images suffisent amplement)
np.random.seed(42)
indices = np.random.choice(len(X_filtered), 5000, replace=False)
X_sub = X_filtered[indices]
y_sub = y_filtered[indices]

# Découpage Train (80%) / Test (20%)
X_train, X_test, y_train, y_test = train_test_split(X_sub, y_sub, test_size=0.20, random_state=42)

# Normalisation des pixels (Standardisation)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Données prêtes ! Train set : {X_train_scaled.shape[0]} images, Test set : {X_test_scaled.shape[0]} images.")

# Dictionnaires pour stocker les résultats finaux
results_time = {}
results_acc = {}
matrices_confusion = {}

# -------------------------------------------------------------------------
# 2. MODÈLE CONVEXE : SVM (RBF) VIA ALGORITHME SMO
# -------------------------------------------------------------------------
print("\n2. Entraînement du SVM (Optimisation Convexe via SMO)...")
start_time = time.time()
# On utilise un noyau RBF, la bibliothèque sklearn utilise l'algorithme SMO en arrière-plan
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train_scaled, y_train)
svm_duration = time.time() - start_time

y_pred_svm = svm_model.predict(X_test_scaled)
svm_acc = accuracy_score(y_test, y_pred_svm)

results_time['SVM (RBF/SMO)'] = svm_duration
results_acc['SVM (RBF/SMO)'] = svm_acc
matrices_confusion['SVM (RBF/SMO)'] = confusion_matrix(y_test, y_pred_svm)
print(f"SVM entraîné en {svm_duration:.3f}s | Accuracy Test : {svm_acc*100:.2f}%")

# -------------------------------------------------------------------------
# 3. MODÈLE NON CONVEXE : MLP (RÉSEAU DE NEURONES EN PYTORCH)
# -------------------------------------------------------------------------
# Configuration PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Conversion des données en Tenseurs PyTorch
X_train_t = torch.FloatTensor(X_train_scaled).to(device)
y_train_t = torch.FloatTensor(y_train).view(-1, 1).to(device)
X_test_t = torch.FloatTensor(X_test_scaled).to(device)
y_test_t = torch.FloatTensor(y_test).view(-1, 1).to(device)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)

# Définition de l'architecture du MLP (784 entrées -> 128 cachés -> 1 sortie)
# Ajout de Dropout pour régulariser et éviter l'overfitting
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Dropout(0.3),  # Régularisation : désactive 30% des neurones aléatoirement
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

# Fonction d'entraînement du MLP générique selon l'optimiseur choisi
# Inclut : Weight Decay (L2) + Early Stopping pour éviter l'overfitting
def train_mlp(optimizer_name, opt_func, lr=0.001, epochs=40, patience=7):
    model = MLP().to(device)
    criterion = nn.BCELoss() # Binary Cross Entropy Loss pour classification binaire
    
    # Weight Decay (régularisation L2) ajouté à tous les optimiseurs
    if optimizer_name == 'MLP - SGD (Momentum)':
        optimizer = opt_func(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = opt_func(model.parameters(), lr=lr, weight_decay=1e-4)
        
    history_loss_train, history_loss_test = [], []
    history_acc_train, history_acc_test = [], []
    
    # Early Stopping : on sauvegarde le meilleur modèle
    best_loss_test = float('inf')
    best_model_state = None
    epochs_without_improvement = 0
    
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
        # Évaluation à chaque époque pour tracer les courbes
        model.eval()
        with torch.no_grad():
            train_outputs = model(X_train_t)
            test_outputs = model(X_test_t)
            
            loss_train = criterion(train_outputs, y_train_t).item()
            loss_test = criterion(test_outputs, y_test_t).item()
            
            acc_train = accuracy_score(y_train, (train_outputs.cpu().numpy() > 0.5).astype(int))
            acc_test = accuracy_score(y_test, (test_outputs.cpu().numpy() > 0.5).astype(int))
            
            history_loss_train.append(loss_train)
            history_loss_test.append(loss_test)
            history_acc_train.append(acc_train)
            history_acc_test.append(acc_test)
            
            # Early Stopping : vérifier si la loss test s'améliore
            if loss_test < best_loss_test:
                best_loss_test = loss_test
                best_model_state = model.state_dict().copy()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                
            if epochs_without_improvement >= patience:
                print(f"   [Early Stopping] Arret a l'epoque {epoch+1} (pas d'amelioration depuis {patience} epoques)")
                break
            
    duration = time.time() - start_time
    
    # Restaurer le meilleur modèle (celui avec la loss test la plus basse)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Prédictions finales sur le Test Set avec le meilleur modèle
    model.eval()
    with torch.no_grad():
        final_preds = (model(X_test_t).cpu().numpy() > 0.5).astype(int)
        final_acc = accuracy_score(y_test, final_preds)
    
    return duration, final_acc, final_preds, history_loss_train, history_loss_test, history_acc_train, history_acc_test

# Entraînement avec SGD + Momentum
print("\n3a. Entraînement du MLP avec SGD (Momentum)...")
sgd_res = train_mlp('MLP - SGD (Momentum)', optim.SGD, lr=0.01)
results_time['MLP - SGD (Momentum)'] = sgd_res[0]
results_acc['MLP - SGD (Momentum)'] = sgd_res[1]
matrices_confusion['MLP - SGD (Momentum)'] = confusion_matrix(y_test, sgd_res[2])

# Entraînement avec Adam
print("3b. Entraînement du MLP avec Adam (Gradient Adaptatif)...")
adam_res = train_mlp('MLP - Adam', optim.Adam, lr=0.001)
results_time['MLP - Adam'] = adam_res[0]
results_acc['MLP - Adam'] = adam_res[1]
matrices_confusion['MLP - Adam'] = confusion_matrix(y_test, adam_res[2])

# -------------------------------------------------------------------------
# 4. TRACÉ DES GRAPHES ET AFFICHAGE DES RÉSULTATS
# -------------------------------------------------------------------------
print("\n4. Génération des graphiques comparatifs...")

# Graphe 1 : Courbes d'apprentissage Loss et Accuracy (SGD vs Adam)
sgd_epochs = range(1, len(sgd_res[3]) + 1)
adam_epochs = range(1, len(adam_res[3]) + 1)
fig, ax = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Comparaison SGD (Momentum) vs Adam - Entraînement du MLP", fontsize=14, fontweight='bold', y=1.02)

# Subplot 1: Loss
ax[0].plot(sgd_epochs, sgd_res[3], 'r-', linewidth=2, label='SGD - Train')
ax[0].plot(sgd_epochs, sgd_res[4], 'r--', linewidth=2, label='SGD - Test')
ax[0].plot(adam_epochs, adam_res[3], 'b-', linewidth=2, label='Adam - Train')
ax[0].plot(adam_epochs, adam_res[4], 'b--', linewidth=2, label='Adam - Test')
ax[0].set_title("Evolution de la Loss au fil des epoques")
ax[0].set_xlabel("Epoque")
ax[0].set_ylabel("Loss (Binary Cross-Entropy)")
ax[0].grid(True, alpha=0.3)
ax[0].legend()

# Subplot 2: Accuracy
ax[1].plot(sgd_epochs, sgd_res[5], 'r-', linewidth=2, label='SGD - Train')
ax[1].plot(sgd_epochs, sgd_res[6], 'r--', linewidth=2, label='SGD - Test')
ax[1].plot(adam_epochs, adam_res[5], 'b-', linewidth=2, label='Adam - Train')
ax[1].plot(adam_epochs, adam_res[6], 'b--', linewidth=2, label='Adam - Test')
ax[1].set_title("Evolution de l'Accuracy au fil des epoques")
ax[1].set_xlabel("Epoque")
ax[1].set_ylabel("Accuracy")
ax[1].grid(True, alpha=0.3)
ax[1].legend()
plt.tight_layout()
plt.savefig('learning_curves.png', dpi=150, bbox_inches='tight')
print("   -> learning_curves.png sauvegarde")
plt.show()

# Graphe 2 : Affichage des Matrices de Confusion
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Matrices de Confusion sur le Test Set (Chiffres 3 vs 8)", fontsize=14, fontweight='bold', y=1.05)

for i, (model_name, cm) in enumerate(matrices_confusion.items()):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Chiffre 3', 'Chiffre 8'])
    disp.plot(ax=axes[i], cmap='Blues', values_format='d')
    axes[i].set_title(model_name, fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
print("   -> confusion_matrices.png sauvegarde")
plt.show()

# Graphe 3 : Barres comparatives (Accuracy + Temps)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Comparaison Globale des Methodes", fontsize=14, fontweight='bold')

colors = ['#2ecc71', '#e74c3c', '#3498db']
methods = list(results_acc.keys())

# Barres Accuracy
bars1 = ax1.bar(methods, [v*100 for v in results_acc.values()], color=colors, edgecolor='black', linewidth=0.5)
ax1.set_title("Accuracy sur le Test Set (%)")
ax1.set_ylabel("Accuracy (%)")
ax1.set_ylim(95, 100)
for bar, val in zip(bars1, results_acc.values()):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1, f'{val*100:.2f}%', ha='center', fontweight='bold')

# Barres Temps
bars2 = ax2.bar(methods, results_time.values(), color=colors, edgecolor='black', linewidth=0.5)
ax2.set_title("Temps d'entrainement (secondes)")
ax2.set_ylabel("Temps (s)")
for bar, val in zip(bars2, results_time.values()):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1, f'{val:.2f}s', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('comparison_bars.png', dpi=150, bbox_inches='tight')
print("   -> comparison_bars.png sauvegarde")
plt.show()

# Graphe 4 : Loss Train vs Test separees pour chaque optimiseur
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Analyse du Gap Train/Test par Optimiseur (Regularisation)", fontsize=14, fontweight='bold', y=1.02)

# SGD
ax1.plot(sgd_epochs, sgd_res[3], 'r-', linewidth=2, label='Train Loss')
ax1.plot(sgd_epochs, sgd_res[4], 'r--', linewidth=2, label='Test Loss')
ax1.fill_between(sgd_epochs, sgd_res[3], sgd_res[4], alpha=0.15, color='red', label='Gap Train/Test')
ax1.axvline(x=len(sgd_res[3]), color='gray', linestyle=':', linewidth=1.5, label=f'Early Stop (ep. {len(sgd_res[3])})')
ax1.set_title("SGD + Momentum", fontsize=12, fontweight='bold')
ax1.set_xlabel("Epoque")
ax1.set_ylabel("Loss (BCE)")
ax1.grid(True, alpha=0.3)
ax1.legend()

# Adam
ax2.plot(adam_epochs, adam_res[3], 'b-', linewidth=2, label='Train Loss')
ax2.plot(adam_epochs, adam_res[4], 'b--', linewidth=2, label='Test Loss')
ax2.fill_between(adam_epochs, adam_res[3], adam_res[4], alpha=0.15, color='blue', label='Gap Train/Test')
ax2.axvline(x=len(adam_res[3]), color='gray', linestyle=':', linewidth=1.5, label=f'Early Stop (ep. {len(adam_res[3])})')
ax2.set_title("Adam", fontsize=12, fontweight='bold')
ax2.set_xlabel("Epoque")
ax2.set_ylabel("Loss (BCE)")
ax2.grid(True, alpha=0.3)
ax2.legend()
plt.tight_layout()
plt.savefig('train_test_gap.png', dpi=150, bbox_inches='tight')
print("   -> train_test_gap.png sauvegarde")
plt.show()

# 5. Tableau récapitulatif final
df_results = pd.DataFrame({
    'Temps d\'entrainement (s)': [results_time['SVM (RBF/SMO)'], results_time['MLP - SGD (Momentum)'], results_time['MLP - Adam']],
    'Accuracy (Test Set)': [results_acc['SVM (RBF/SMO)'], results_acc['MLP - SGD (Momentum)'], results_acc['MLP - Adam']]
}, index=['SVM (RBF/SMO)', 'MLP - SGD (Momentum)', 'MLP - Adam'])

print("\n=== TABLEAU COMPARATIF FINAL ===")
print(df_results.to_string())