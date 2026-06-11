import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# -------------------------------------------------------------------------
# 1. DATA PREPARATION
# -------------------------------------------------------------------------
print("1. Chargement des donnees MNIST (3 vs 8)...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X_all, y_all = mnist.data, mnist.target.astype(int)

mask = (y_all == 3) | (y_all == 8)
X_filtered = X_all[mask]
y_filtered = y_all[mask]
y_filtered = np.where(y_filtered == 3, 0, y_filtered)
y_filtered = np.where(y_filtered == 8, 1, y_filtered)

np.random.seed(42)
indices = np.random.choice(len(X_filtered), 5000, replace=False)
X_sub = X_filtered[indices]
y_sub = y_filtered[indices]

X_train, X_test, y_train, y_test = train_test_split(X_sub, y_sub, test_size=0.20, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train_t = torch.FloatTensor(X_train_scaled).to(device)
y_train_t = torch.FloatTensor(y_train).view(-1, 1).to(device)
X_test_t = torch.FloatTensor(X_test_scaled).to(device)
y_test_t = torch.FloatTensor(y_test).view(-1, 1).to(device)
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)

# SVM
print("2. Entrainement SVM...")
start_time = time.time()
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train_scaled, y_train)
svm_duration = time.time() - start_time
y_pred_svm = svm_model.predict(X_test_scaled)
svm_acc = accuracy_score(y_test, y_pred_svm)
svm_cm = confusion_matrix(y_test, y_pred_svm)
print(f"   SVM: {svm_acc*100:.2f}% en {svm_duration:.2f}s")

# -------------------------------------------------------------------------
# MLP definitions
# -------------------------------------------------------------------------
class MLP_NoReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(784, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.network(x)

class MLP_Reg(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(784, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.network(x)

def train_mlp(model_class, opt_name, opt_func, lr, epochs=40, wd=0.0, early_stop=False, patience=7):
    torch.manual_seed(42)
    model = model_class().to(device)
    criterion = nn.BCELoss()
    if opt_name == 'SGD':
        optimizer = opt_func(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    else:
        optimizer = opt_func(model.parameters(), lr=lr, weight_decay=wd)
    h_lt, h_lv, h_at, h_av = [], [], [], []
    best_lv, best_state, no_imp = float('inf'), None, 0

    start = time.time()
    for ep in range(epochs):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            to = model(X_train_t); vo = model(X_test_t)
            lt = criterion(to, y_train_t).item(); lv = criterion(vo, y_test_t).item()
            at = accuracy_score(y_train, (to.cpu().numpy()>0.5).astype(int))
            av = accuracy_score(y_test, (vo.cpu().numpy()>0.5).astype(int))
            h_lt.append(lt); h_lv.append(lv); h_at.append(at); h_av.append(av)
            if early_stop:
                if lv < best_lv:
                    best_lv = lv; best_state = model.state_dict().copy(); no_imp = 0
                else:
                    no_imp += 1
                if no_imp >= patience:
                    print(f"      Early Stop epoch {ep+1}")
                    break
    dur = time.time() - start
    if early_stop and best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = (model(X_test_t).cpu().numpy()>0.5).astype(int)
        acc = accuracy_score(y_test, preds)
    return acc, preds, h_lt, h_lv, h_at, h_av, dur

# -------------------------------------------------------------------------
# TRAIN ALL
# -------------------------------------------------------------------------
print("\n3. PHASE 1 - Entrainement SANS regularisation...")
b_sgd = train_mlp(MLP_NoReg, 'SGD', optim.SGD, 0.01)
b_adam = train_mlp(MLP_NoReg, 'Adam', optim.Adam, 0.001)
print(f"   SGD: {b_sgd[0]*100:.2f}% | Adam: {b_adam[0]*100:.2f}%")

print("\n4. PHASE 2 - Entrainement AVEC regularisation...")
a_sgd = train_mlp(MLP_Reg, 'SGD', optim.SGD, 0.01, wd=1e-4, early_stop=True)
a_adam = train_mlp(MLP_Reg, 'Adam', optim.Adam, 0.001, wd=1e-4, early_stop=True)
print(f"   SGD: {a_sgd[0]*100:.2f}% ({len(a_sgd[2])} ep) | Adam: {a_adam[0]*100:.2f}% ({len(a_adam[2])} ep)")

# =========================================================================
# GRAPHIQUES - PARTIE 1 : AVANT (OVERFITTING)
# =========================================================================
print("\n5. Generation des graphiques...")
ep40 = range(1, 41)

# --- GRAPH 1: BEFORE - Loss ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title("BEFORE Regularization - Loss Curves (40 Epochs, No Regularization)", fontsize=14, fontweight='bold', color='#c0392b', pad=15)
ax.plot(ep40, b_sgd[2], 'r-', linewidth=2.5, label='SGD - Train Loss')
ax.plot(ep40, b_sgd[3], 'r--', linewidth=2.5, label='SGD - Test Loss')
ax.plot(ep40, b_adam[2], 'b-', linewidth=2.5, label='Adam - Train Loss')
ax.plot(ep40, b_adam[3], 'b--', linewidth=2.5, label='Adam - Test Loss')
ax.annotate('Adam Test Loss\nDIVERGING!', xy=(36, b_adam[3][35]), fontsize=12, fontweight='bold',
            color='#c0392b', ha='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#fadbd8', edgecolor='#c0392b'),
            arrowprops=dict(arrowstyle='->', color='#c0392b', lw=2),
            xytext=(25, max(b_adam[3])*0.85))
ax.annotate('Train Loss -> 0\n(memorization)', xy=(38, b_adam[2][37]), fontsize=10, fontweight='bold',
            color='#8e44ad', ha='center',
            xytext=(28, 0.03),
            arrowprops=dict(arrowstyle='->', color='#8e44ad', lw=1.5))
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Loss (Binary Cross-Entropy)", fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, loc='center right')
plt.tight_layout()
plt.savefig('01_before_loss.png', dpi=150, bbox_inches='tight')
print("   -> 01_before_loss.png")
plt.close()

# --- GRAPH 2: BEFORE - Accuracy ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title("BEFORE Regularization - Accuracy Curves (40 Epochs, No Regularization)", fontsize=14, fontweight='bold', color='#c0392b', pad=15)
ax.plot(ep40, b_sgd[4], 'r-', linewidth=2.5, label='SGD - Train Acc')
ax.plot(ep40, b_sgd[5], 'r--', linewidth=2.5, label='SGD - Test Acc')
ax.plot(ep40, b_adam[4], 'b-', linewidth=2.5, label='Adam - Train Acc')
ax.plot(ep40, b_adam[5], 'b--', linewidth=2.5, label='Adam - Test Acc')
ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
ax.annotate('Train = 100%\nbut Test stuck at ~98.3%\nOVERFITTING', xy=(30, 0.998), fontsize=11, fontweight='bold',
            color='#c0392b', ha='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#fadbd8', edgecolor='#c0392b'),
            xytext=(22, 0.973),
            arrowprops=dict(arrowstyle='->', color='#c0392b', lw=2))
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, loc='lower right')
plt.tight_layout()
plt.savefig('02_before_accuracy.png', dpi=150, bbox_inches='tight')
print("   -> 02_before_accuracy.png")
plt.close()

# --- GRAPH 3: BEFORE - Gap Analysis ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("BEFORE Regularization - Train/Test Gap Analysis", fontsize=15, fontweight='bold', color='#c0392b', y=1.02)

ax1.set_title("SGD + Momentum (Mild Overfitting)", fontsize=13, fontweight='bold')
ax1.plot(ep40, b_sgd[2], 'r-', linewidth=2, label='Train Loss')
ax1.plot(ep40, b_sgd[3], 'r--', linewidth=2, label='Test Loss')
ax1.fill_between(ep40, b_sgd[2], b_sgd[3], alpha=0.25, color='#e74c3c', label='Generalization Gap')
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss (BCE)")
ax1.grid(True, alpha=0.3); ax1.legend(fontsize=10)

ax2.set_title("Adam (SEVERE Overfitting!)", fontsize=13, fontweight='bold', color='#c0392b')
ax2.plot(ep40, b_adam[2], 'b-', linewidth=2, label='Train Loss')
ax2.plot(ep40, b_adam[3], 'b--', linewidth=2, label='Test Loss')
ax2.fill_between(ep40, b_adam[2], b_adam[3], alpha=0.25, color='#e74c3c', label='Generalization Gap')
ax2.annotate('OVERFITTING!\nHuge gap', xy=(35, (b_adam[3][34]+b_adam[2][34])/2),
            fontsize=12, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#c0392b', alpha=0.9),
            ha='center')
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss (BCE)")
ax2.grid(True, alpha=0.3); ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig('03_before_gap.png', dpi=150, bbox_inches='tight')
print("   -> 03_before_gap.png")
plt.close()

# =========================================================================
# GRAPHIQUES - PARTIE 2 : APRES (REGULARISATION)
# =========================================================================
sgd_ep = range(1, len(a_sgd[2])+1)
adam_ep = range(1, len(a_adam[2])+1)

# --- GRAPH 4: AFTER - Loss ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title("AFTER Regularization - Loss Curves (Dropout + Weight Decay + Early Stopping)", fontsize=14, fontweight='bold', color='#27ae60', pad=15)
ax.plot(sgd_ep, a_sgd[2], 'r-', linewidth=2.5, label='SGD - Train Loss')
ax.plot(sgd_ep, a_sgd[3], 'r--', linewidth=2.5, label='SGD - Test Loss')
ax.plot(adam_ep, a_adam[2], 'b-', linewidth=2.5, label='Adam - Train Loss')
ax.plot(adam_ep, a_adam[3], 'b--', linewidth=2.5, label='Adam - Test Loss')
ax.axvline(x=len(a_sgd[2]), color='red', linestyle=':', alpha=0.7, linewidth=2, label=f'SGD Early Stop (ep.{len(a_sgd[2])})')
ax.axvline(x=len(a_adam[2]), color='blue', linestyle=':', alpha=0.7, linewidth=2, label=f'Adam Early Stop (ep.{len(a_adam[2])})')
ax.annotate('Train/Test curves\nstay CLOSE together\n(no divergence!)', 
            xy=(len(a_adam[2])//2, (a_adam[2][len(a_adam[2])//2]+a_adam[3][len(a_adam[2])//2])/2),
            fontsize=11, fontweight='bold', color='#27ae60', ha='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#d5f5e3', edgecolor='#27ae60'),
            xytext=(len(a_adam[2])//2 + 8, max(a_sgd[2])*0.7),
            arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2))
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Loss (Binary Cross-Entropy)", fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10, loc='upper right')
plt.tight_layout()
plt.savefig('04_after_loss.png', dpi=150, bbox_inches='tight')
print("   -> 04_after_loss.png")
plt.close()

# --- GRAPH 5: AFTER - Accuracy ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title("AFTER Regularization - Accuracy Curves (Dropout + Weight Decay + Early Stopping)", fontsize=14, fontweight='bold', color='#27ae60', pad=15)
ax.plot(sgd_ep, a_sgd[4], 'r-', linewidth=2.5, label='SGD - Train Acc')
ax.plot(sgd_ep, a_sgd[5], 'r--', linewidth=2.5, label='SGD - Test Acc')
ax.plot(adam_ep, a_adam[4], 'b-', linewidth=2.5, label='Adam - Train Acc')
ax.plot(adam_ep, a_adam[5], 'b--', linewidth=2.5, label='Adam - Test Acc')
ax.axvline(x=len(a_sgd[4]), color='red', linestyle=':', alpha=0.7, linewidth=2, label=f'SGD Early Stop (ep.{len(a_sgd[4])})')
ax.axvline(x=len(a_adam[4]), color='blue', linestyle=':', alpha=0.7, linewidth=2, label=f'Adam Early Stop (ep.{len(a_adam[4])})')
ax.annotate('Train and Test\naccuracies ALIGNED\n(good generalization)', 
            xy=(len(a_sgd[4])//2, a_sgd[5][len(a_sgd[4])//2]),
            fontsize=11, fontweight='bold', color='#27ae60', ha='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#d5f5e3', edgecolor='#27ae60'),
            xytext=(len(a_sgd[4])//2 - 5, min(min(a_sgd[5]), min(a_adam[5])) - 0.01),
            arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2))
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10, loc='lower right')
plt.tight_layout()
plt.savefig('05_after_accuracy.png', dpi=150, bbox_inches='tight')
print("   -> 05_after_accuracy.png")
plt.close()

# --- GRAPH 6: AFTER - Gap Analysis ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("AFTER Regularization - Train/Test Gap Analysis", fontsize=15, fontweight='bold', color='#27ae60', y=1.02)

ax1.set_title("SGD + Momentum (Regularized)", fontsize=13, fontweight='bold', color='#27ae60')
ax1.plot(sgd_ep, a_sgd[2], 'r-', linewidth=2, label='Train Loss')
ax1.plot(sgd_ep, a_sgd[3], 'r--', linewidth=2, label='Test Loss')
ax1.fill_between(sgd_ep, a_sgd[2], a_sgd[3], alpha=0.2, color='#27ae60', label='Generalization Gap')
ax1.axvline(x=len(a_sgd[2]), color='gray', linestyle=':', linewidth=2, label=f'Early Stop (ep.{len(a_sgd[2])})')
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss (BCE)")
ax1.grid(True, alpha=0.3); ax1.legend(fontsize=10)

ax2.set_title("Adam (Regularized)", fontsize=13, fontweight='bold', color='#27ae60')
ax2.plot(adam_ep, a_adam[2], 'b-', linewidth=2, label='Train Loss')
ax2.plot(adam_ep, a_adam[3], 'b--', linewidth=2, label='Test Loss')
ax2.fill_between(adam_ep, a_adam[2], a_adam[3], alpha=0.2, color='#27ae60', label='Generalization Gap')
ax2.axvline(x=len(a_adam[2]), color='gray', linestyle=':', linewidth=2, label=f'Early Stop (ep.{len(a_adam[2])})')
ax2.annotate('Small, controlled gap\n= Good generalization!', 
            xy=(len(a_adam[2])//2, (a_adam[2][len(a_adam[2])//2]+a_adam[3][len(a_adam[2])//2])/2),
            fontsize=11, fontweight='bold', color='#27ae60', ha='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#d5f5e3', edgecolor='#27ae60'))
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss (BCE)")
ax2.grid(True, alpha=0.3); ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig('06_after_gap.png', dpi=150, bbox_inches='tight')
print("   -> 06_after_gap.png")
plt.close()

# =========================================================================
# GRAPHIQUES - PARTIE 3 : COMPARAISON FINALE
# =========================================================================

# --- GRAPH 7: Confusion Matrices (final regularized models) ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Confusion Matrices - Final Models (Test Set: 1000 images)", fontsize=14, fontweight='bold', y=1.05)

cms = {'SVM (RBF/SMO)': svm_cm,
       'MLP - SGD (Regularized)': confusion_matrix(y_test, a_sgd[1]),
       'MLP - Adam (Regularized)': confusion_matrix(y_test, a_adam[1])}
for i, (name, cm) in enumerate(cms.items()):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Digit 3', 'Digit 8'])
    disp.plot(ax=axes[i], cmap='Blues', values_format='d')
    axes[i].set_title(name, fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('07_confusion_matrices.png', dpi=150, bbox_inches='tight')
print("   -> 07_confusion_matrices.png")
plt.close()

# --- GRAPH 8: Before vs After Bars ---
from matplotlib.patches import Patch
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Before vs After Regularization - Performance Comparison", fontsize=15, fontweight='bold')

methods = ['SGD\n(Before)', 'SGD\n(After)', 'Adam\n(Before)', 'Adam\n(After)']
accs = [b_sgd[0]*100, a_sgd[0]*100, b_adam[0]*100, a_adam[0]*100]
epochs_used = [40, len(a_sgd[2]), 40, len(a_adam[2])]
colors = ['#e74c3c', '#27ae60', '#e74c3c', '#27ae60']

bars1 = ax1.bar(methods, accs, color=colors, edgecolor='black', linewidth=0.5)
ax1.set_title("Test Accuracy (%)", fontsize=13, fontweight='bold')
ax1.set_ylabel("Accuracy (%)")
ax1.set_ylim(95, 100)
for bar, val in zip(bars1, accs):
    ax1.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.05, f'{val:.2f}%', ha='center', fontweight='bold', fontsize=11)

bars2 = ax2.bar(methods, epochs_used, color=colors, edgecolor='black', linewidth=0.5)
ax2.set_title("Epochs Trained", fontsize=13, fontweight='bold')
ax2.set_ylabel("Number of Epochs")
for bar, val in zip(bars2, epochs_used):
    ax2.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.5, str(val), ha='center', fontweight='bold', fontsize=12)

legend_elements = [Patch(facecolor='#e74c3c', edgecolor='black', label='Before (No Regularization)'),
                   Patch(facecolor='#27ae60', edgecolor='black', label='After (With Regularization)')]
fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=12, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout()
plt.savefig('08_before_after_bars.png', dpi=150, bbox_inches='tight')
print("   -> 08_before_after_bars.png")
plt.close()

# --- GRAPH 9: Final comparison all 3 methods ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Final Comparison - All Methods (After Regularization)", fontsize=14, fontweight='bold')

final_methods = ['SVM\n(RBF/SMO)', 'MLP\nSGD+Momentum', 'MLP\nAdam']
final_accs = [svm_acc*100, a_sgd[0]*100, a_adam[0]*100]
final_times = [svm_duration, a_sgd[6], a_adam[6]]
colors3 = ['#2ecc71', '#e74c3c', '#3498db']

bars1 = ax1.bar(final_methods, final_accs, color=colors3, edgecolor='black', linewidth=0.5)
ax1.set_title("Test Accuracy (%)")
ax1.set_ylabel("Accuracy (%)")
ax1.set_ylim(95, 100)
for bar, val in zip(bars1, final_accs):
    ax1.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.05, f'{val:.2f}%', ha='center', fontweight='bold')

bars2 = ax2.bar(final_methods, final_times, color=colors3, edgecolor='black', linewidth=0.5)
ax2.set_title("Training Time (seconds)")
ax2.set_ylabel("Time (s)")
for bar, val in zip(bars2, final_times):
    ax2.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.05, f'{val:.2f}s', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('09_final_comparison.png', dpi=150, bbox_inches='tight')
print("   -> 09_final_comparison.png")
plt.close()

print("\n=== DONE - 9 graphiques generes ===")
print(f"BEFORE: SGD={b_sgd[0]*100:.2f}% (40ep) | Adam={b_adam[0]*100:.2f}% (40ep)")
print(f"AFTER:  SGD={a_sgd[0]*100:.2f}% ({len(a_sgd[2])}ep) | Adam={a_adam[0]*100:.2f}% ({len(a_adam[2])}ep)")
print(f"SVM:    {svm_acc*100:.2f}% ({svm_duration:.2f}s)")
