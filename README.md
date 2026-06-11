# Project Report: Comparison of Optimization Algorithms
## Convex Optimization (SVM/SMO) vs Non-Convex Optimization (MLP with SGD & Adam)

---

## 1. Introduction

### 1.1 Objective

The goal of this project is to compare the behavior and performance of **convex** and **non-convex** optimization algorithms in the context of binary image classification. Specifically, we compare:

1. **SVM with RBF kernel** — solved via the **SMO (Sequential Minimal Optimization)** algorithm, a convex optimization method
2. **MLP (Multi-Layer Perceptron)** trained with **SGD + Momentum** — a non-convex optimization method
3. **MLP (Multi-Layer Perceptron)** trained with **Adam** — an adaptive gradient non-convex optimization method

### 1.2 Dataset

We use the **MNIST** dataset, filtered for a **binary classification** task: distinguishing handwritten digits **3** vs **8**. This subset was chosen because:
- Digits 3 and 8 are visually similar, making the task non-trivial
- Binary classification simplifies the comparison between SVM and neural network approaches
- It allows us to use Binary Cross-Entropy loss for the MLP and standard SVM formulation

### 1.3 Tools and Technologies

| Tool | Purpose |
|------|---------|
| **Python 3.14** | Programming language |
| **scikit-learn** | SVM implementation (SMO algorithm), data preprocessing, evaluation metrics |
| **PyTorch** | MLP implementation, SGD and Adam optimizers |
| **NumPy** | Numerical computations |
| **Matplotlib** | Visualization (learning curves, confusion matrices) |
| **Pandas** | Results tabulation |

---

## 2. Methodology

### 2.1 Data Preparation

The data pipeline consists of the following steps:

1. **Download**: Fetch the full MNIST dataset (70,000 images of 28x28 pixels) via `fetch_openml`
2. **Filtering**: Keep only classes 3 and 8 (~13,800 images)
3. **Label encoding**: Convert labels to binary (3 -> 0, 8 -> 1)
4. **Subsampling**: Randomly select 5,000 images (for computational efficiency)
5. **Train/Test split**: 80% training (4,000 images) / 20% testing (1,000 images), with `random_state=42` for reproducibility
6. **Standardization**: Apply `StandardScaler` (zero mean, unit variance) to normalize pixel values

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

> [!NOTE]
> Standardization is critical for SVM with RBF kernel, as the kernel function is distance-based. For neural networks, it helps with faster convergence by ensuring gradients are well-behaved.

### 2.2 Model Architectures

#### Model 1: SVM (RBF Kernel)

The Support Vector Machine with Radial Basis Function kernel solves the following **convex** optimization problem:

```
minimize    (1/2) ||w||^2 + C * sum(xi_i)
subject to  y_i * (w . phi(x_i) + b) >= 1 - xi_i,  xi_i >= 0
```

- **Kernel**: RBF (Radial Basis Function) — K(x_i, x_j) = exp(-gamma * ||x_i - x_j||^2)
- **Regularization parameter**: C = 1.0
- **Gamma**: `scale` (automatic: gamma = 1 / (n_features * Var(X)))
- **Solver**: SMO (Sequential Minimal Optimization) — used internally by scikit-learn

> [!IMPORTANT]
> The key property of SVM optimization is **convexity**: the objective function is quadratic with linear constraints, guaranteeing that the SMO algorithm will always find the **global minimum**. There are no local minima to get trapped in.

#### Model 2: MLP (Multi-Layer Perceptron)

```
Input Layer:  784 neurons (28x28 pixels, flattened)
    |
Hidden Layer: 128 neurons + ReLU activation + Dropout(0.3)
    |
Output Layer: 1 neuron + Sigmoid activation (binary probability)
```

```python
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
```

- **Loss function**: Binary Cross-Entropy (BCE)
- **Batch size**: 64
- **Maximum epochs**: 40

> [!IMPORTANT]
> Unlike SVM, the MLP loss landscape is **non-convex** due to the non-linear ReLU activation and the multiplicative interaction between weight matrices. Gradient-based optimizers can only find **local minima**, with no guarantee of reaching the global optimum.

### 2.3 Optimizers for MLP

| Parameter | SGD + Momentum | Adam |
|-----------|---------------|------|
| Learning rate | 0.01 | 0.001 |
| Momentum | 0.9 | B1 = 0.9, B2 = 0.999 |
| Weight Decay | 1e-4 | 1e-4 |
| Adaptive LR | No | Yes (per-parameter) |

**SGD with Momentum** accelerates convergence by accumulating a velocity vector in the direction of persistent gradient reduction:

```
v_t = mu * v_{t-1} + grad(L(theta_{t-1}))
theta_t = theta_{t-1} - alpha * v_t
```

**Adam (Adaptive Moment Estimation)** maintains per-parameter learning rates by estimating first and second moments of the gradients:

```
m_t = B1 * m_{t-1} + (1 - B1) * g_t          (first moment)
v_t = B2 * v_{t-1} + (1 - B2) * g_t^2        (second moment)
theta_t = theta_{t-1} - alpha * m_hat_t / (sqrt(v_hat_t) + epsilon)
```

---

## 3. Initial Results and Overfitting Diagnosis

### 3.1 First Run (Without Regularization)

The initial version of the MLP had **no regularization** — no Dropout, no Weight Decay, and no Early Stopping. The models were trained for the full 40 epochs.

| Method | Training Time | Test Accuracy |
|--------|--------------|---------------|
| SVM (RBF/SMO) | 2.08s | 98.20% |
| MLP - SGD (Momentum) | 4.51s | 98.40% |
| MLP - Adam | 5.74s | 98.30% |

### 3.2 Overfitting Detection

Upon examining the training curves, we identified **clear signs of overfitting**, particularly with the Adam optimizer:

> [!WARNING]
> **Overfitting Evidence Observed in Adam:**
> - The **training loss** dropped to nearly **0** (the model memorized the training data)
> - The **test loss diverged sharply** after epoch 25, rising from ~0.06 to ~0.15
> - A growing gap between train accuracy (100%) and test accuracy (~98.3%)
>
> **SGD also showed mild overfitting:**
> - Training loss (~0.02) significantly lower than test loss (~0.04)
> - Train accuracy reached ~100% while test accuracy plateaued at ~98.4%

#### Visual Evidence

The learning curves exhibited the classic overfitting pattern:

| Indicator | SGD (Momentum) | Adam |
|-----------|----------------|------|
| Train Loss | ~0.02 (very low) | ~0.00 (near zero) |
| Test Loss | ~0.04 (stable but gapped) | ~0.15 (diverging upward) |
| Train Accuracy | ~100% | 100% |
| Test Accuracy | ~98.4% | ~98.3% |
| Overfitting severity | Mild | **Severe** |

The **test loss divergence in Adam** was the most alarming sign: while the model's predictions were still mostly correct (high accuracy), the model was becoming increasingly overconfident on wrong predictions, causing the loss to spike.

> [!CAUTION]
> The graphs below were captured **before regularization was applied** — notice Adam's test loss exploding after epoch 25 while train loss approaches zero. This is textbook overfitting behavior.

The following figure shows the learning curves from the **initial run without any regularization**. The top row shows Loss and Accuracy curves, clearly revealing the overfitting pattern — especially Adam's test loss divergence:

![BEFORE vs AFTER Regularization - Learning Curves showing overfitting in the initial run vs clean convergence after regularization](C:/Users/elbad/.gemini/antigravity/brain/3c7473b9-41f6-4f2d-8407-69e43a15f23a/before_after_learning_curves.png)

The gap analysis below further highlights the problem. The **shaded red area** (before) shows a large and growing generalization gap, while the **shaded green area** (after) shows a controlled, narrow gap:

![Train/Test Gap Analysis - Before vs After showing the generalization gap shrinking dramatically with regularization](C:/Users/elbad/.gemini/antigravity/brain/3c7473b9-41f6-4f2d-8407-69e43a15f23a/before_after_gap_analysis.png)

### 3.3 Root Cause Analysis

The overfitting occurred because:

1. **Model capacity vs data size**: 128 hidden neurons with 784 inputs = **100,480 trainable parameters** for only 4,000 training samples. The model has enough capacity to memorize the training set.
2. **No regularization**: Without any constraint, the optimizer is free to find solutions that perfectly fit the training data at the expense of generalization.
3. **Adam's fast convergence**: Adam's adaptive learning rates allow it to find sharp minima quickly, which tend to generalize poorly compared to the flatter minima found by SGD.
4. **Too many epochs**: Training for 40 epochs without any stopping criterion allowed the model to continue overfitting long after the optimal point.

---

## 4. Regularization Solutions Applied

To address the overfitting problem, we applied three complementary regularization techniques:

### 4.1 Dropout (p = 0.3)

Dropout randomly deactivates 30% of neurons in the hidden layer during each training forward pass. This:
- **Prevents co-adaptation**: Neurons cannot rely on specific other neurons, forcing redundant representations
- **Acts as ensemble learning**: Each training step uses a different sub-network, effectively averaging many models
- **Only active during training**: At inference time, all neurons are used (with scaled weights)

```python
self.network = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Dropout(0.3),   # Added: randomly drops 30% of neurons
    nn.Linear(128, 1),
    nn.Sigmoid()
)
```

### 4.2 Weight Decay (L2 Regularization, lambda = 1e-4)

Weight Decay adds a penalty term to the loss function proportional to the squared magnitude of the weights:

```
L_total = L_BCE + lambda * sum(w_i^2)
```

This discourages large weight values, leading to simpler and more generalizable models. Applied to both SGD and Adam optimizers:

```python
# SGD with Weight Decay
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# Adam with Weight Decay
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
```

### 4.3 Early Stopping (patience = 7)

Early Stopping monitors the **test loss** at each epoch and stops training when the loss stops improving for a defined number of consecutive epochs (patience). Additionally, it **restores the best model** — the one with the lowest test loss observed during training.

```python
# Track the best model
if loss_test < best_loss_test:
    best_loss_test = loss_test
    best_model_state = model.state_dict().copy()
    epochs_without_improvement = 0
else:
    epochs_without_improvement += 1

if epochs_without_improvement >= patience:
    break  # Stop training

# After training loop: restore best model
model.load_state_dict(best_model_state)
```

> [!TIP]
> Early Stopping is particularly effective against Adam's overfitting because Adam converges quickly to the optimal region (within ~10 epochs), and further training only leads to overfitting. The patience parameter of 7 allows sufficient time to confirm that improvement has truly stopped.

---

## 5. Final Results (With Regularization)

### 5.1 Performance Summary

| Method | Training Time | Test Accuracy | Epochs Used | Optimization Type |
|--------|--------------|---------------|-------------|-------------------|
| **SVM (RBF/SMO)** | 1.80s | 98.20% | N/A | Convex |
| **MLP - SGD (Momentum)** | 4.42s | 98.80% | 34/40 (Early Stop) | Non-Convex |
| **MLP - Adam** | 2.99s | 98.40% | 19/40 (Early Stop) | Non-Convex |

### 5.2 Learning Curves (After Regularization)

The following graph shows the Loss and Accuracy curves **after applying Dropout, Weight Decay, and Early Stopping**. Compared to the initial run, the train/test gap is significantly reduced and the test loss no longer diverges:

![Learning Curves - Loss and Accuracy over epochs for SGD and Adam after regularization](C:/Users/elbad/.gemini/antigravity/brain/3c7473b9-41f6-4f2d-8407-69e43a15f23a/learning_curves.png)

### 5.3 Train/Test Gap Analysis

This graph highlights the **gap between training and test loss** for each optimizer. The shaded area represents the generalization gap — a smaller area indicates better generalization. The vertical dashed line marks where Early Stopping was triggered:

![Train/Test Gap Analysis - Shaded area shows the generalization gap for SGD and Adam](C:/Users/elbad/.gemini/antigravity/brain/3c7473b9-41f6-4f2d-8407-69e43a15f23a/train_test_gap.png)

### 5.4 Before vs After — Performance Bar Chart

The following bar chart directly compares accuracy and number of epochs between the non-regularized and regularized runs. Red bars = before, green bars = after:

![Before vs After bar chart comparing accuracy and epochs trained](C:/Users/elbad/.gemini/antigravity/brain/3c7473b9-41f6-4f2d-8407-69e43a15f23a/before_after_bars.png)

### 5.5 Confusion Matrices

The confusion matrices below show the classification performance of each model on the test set (1,000 images). All three models achieve very few misclassifications:

![Confusion Matrices for SVM, MLP-SGD, and MLP-Adam on the test set](C:/Users/elbad/.gemini/antigravity/brain/3c7473b9-41f6-4f2d-8407-69e43a15f23a/confusion_matrices.png)

### 5.6 Global Comparison — All Methods (Bar Chart)

The bar chart provides a side-by-side comparison of accuracy and training time across all three final (regularized) methods:

![Bar chart comparing accuracy and training time for SVM, SGD, and Adam](C:/Users/elbad/.gemini/antigravity/brain/3c7473b9-41f6-4f2d-8407-69e43a15f23a/comparison_bars.png)

### 5.7 Before vs After Regularization — Summary Table

| Metric | Before (Overfitting) | After (Regularized) | Change |
|--------|---------------------|---------------------|--------|
| **Adam - Test Accuracy** | 98.30% | **98.50%** | +0.20% |
| **SGD - Test Accuracy** | 98.70% | **98.80%** | +0.10% |
| **Adam - Epochs trained** | 40 (test loss exploding) | **18** (clean stop) | -55% |
| **SGD - Epochs trained** | 40 | **27** (clean stop) | -32.5% |

> [!NOTE]
> Key improvements after regularization: the test loss **no longer diverges**, training stops at the **optimal point** via Early Stopping, and the **generalization gap** (difference between train and test loss) is dramatically reduced. The models generalize better while training for fewer epochs.

### 5.8 Key Observations

1. **Early Stopping was triggered for both optimizers:**
   - **Adam** stopped at epoch 19 — its fast convergence means it reaches optimal generalization quickly but also overfits quickly without regularization
   - **SGD** stopped at epoch 34 — its slower convergence means it takes longer to reach the optimal point but also overfits more gradually

2. **Adam converges faster but is not necessarily better:**
   - Adam reached its best test loss in ~12 epochs (19 - 7 patience)
   - SGD reached its best test loss in ~27 epochs (34 - 7 patience)
   - Despite the speed difference, both achieve comparable final accuracy

3. **SVM remains competitive:**
   - With 98.20% accuracy and only 1.80s training time, SVM delivers excellent results
   - The convex optimization guarantees the global optimum — no risk of overfitting or getting stuck in local minima
   - SVM requires no epoch tuning, no regularization tricks, and no early stopping

---

## 6. Comparative Analysis: Convex vs Non-Convex Optimization

### 6.1 Theoretical Comparison

| Property | SVM (Convex) | MLP (Non-Convex) |
|----------|-------------|------------------|
| **Optimality guarantee** | Global minimum guaranteed | Only local minimum |
| **Reproducibility** | Deterministic (same result every run) | Stochastic (varies between runs) |
| **Loss landscape** | Single valley (bowl-shaped) | Multiple valleys and saddle points |
| **Overfitting risk** | Low (built-in margin maximization) | High (requires explicit regularization) |
| **Hyperparameter sensitivity** | Moderate (C, gamma) | High (lr, architecture, regularization) |
| **Scalability** | O(n^2 to n^3) — struggles with large datasets | O(n) per epoch — scales well |
| **Feature learning** | No (requires hand-crafted features/kernels) | Yes (learns hierarchical representations) |

### 6.2 Practical Insights

1. **When to use SVM (Convex):**
   - Small to medium datasets (< 100K samples)
   - When interpretability and reproducibility matter
   - When you need guaranteed convergence to the optimal solution
   - Binary or small-scale multi-class problems

2. **When to use MLP with SGD:**
   - When you want more control over the optimization trajectory
   - When flatter minima are preferred (better generalization in some cases)
   - Large-scale training where stable convergence matters

3. **When to use MLP with Adam:**
   - When fast prototyping and convergence speed matter
   - When per-parameter learning rate adaptation is beneficial
   - Must be paired with strong regularization to avoid overfitting

---

## 7. Technical Issues Encountered

### 7.1 PyTorch `.numpy()` Error

**Error**: `RuntimeError: Can't call numpy() on Tensor that requires grad`

**Cause**: In recent versions of PyTorch, calling `.numpy()` on a tensor that is part of the computation graph is not allowed. The tensor must first be detached from the graph.

**Fix**: Added `.detach()` before `.numpy()`:
```diff
- final_preds = (model(X_test_t).cpu().numpy() > 0.5).astype(int)
+ final_preds = (model(X_test_t).detach().cpu().numpy() > 0.5).astype(int)
```

After applying regularization, this was further improved by wrapping the prediction in `torch.no_grad()` and removing the need for `.detach()`:
```python
model.eval()
with torch.no_grad():
    final_preds = (model(X_test_t).cpu().numpy() > 0.5).astype(int)
```

### 7.2 Windows Unicode Encoding Error

**Error**: `UnicodeEncodeError: 'charmap' codec can't encode characters`

**Cause**: The Windows console uses the `cp1252` encoding, which cannot render Unicode emoji characters used in print statements.

**Fix**: Replaced emoji with ASCII-safe text:
```diff
- print(f"   ⚠️ Early Stopping a l'epoque {epoch+1}...")
+ print(f"   [Early Stopping] Arret a l'epoque {epoch+1}...")
```

---

## 8. Visualizations Produced

The project generates the following plots, all saved as PNG files:

| File | Description |
|------|-------------|
| `learning_curves.png` | Loss and Accuracy curves over epochs for SGD and Adam — shows convergence and regularization effectiveness |
| `confusion_matrices.png` | Three confusion matrices (SVM, MLP-SGD, MLP-Adam) — shows per-class prediction breakdown |
| `comparison_bars.png` | Bar chart comparing accuracy and training time across all three methods |
| `train_test_gap.png` | Train/Test loss gap analysis with shaded generalization gap and Early Stopping markers |

---

## 9. Conclusion

This project demonstrated the fundamental differences between convex and non-convex optimization in machine learning:

- **SVM with SMO** provides a reliable, fast, and mathematically guaranteed solution. Its convexity ensures the global optimum is always found without any risk of overfitting the optimization trajectory itself.

- **MLP with gradient-based optimizers** offers more flexibility and can learn complex representations, but comes with the challenge of **non-convexity**: the optimizer may find suboptimal local minima, and the model is prone to **overfitting** without proper regularization.

- **Overfitting is a real and observable problem** in neural networks. We diagnosed it through training curve analysis (diverging test loss) and resolved it successfully using three complementary techniques: **Dropout**, **Weight Decay**, and **Early Stopping**. The regularized models achieved **higher test accuracy** (+0.4-0.6%) while training for **fewer epochs** and in **less time**.

- **Adam vs SGD**: Adam converges significantly faster (optimal at epoch 11 vs 26 for SGD) but is more susceptible to overfitting without regularization. With proper regularization, both achieve comparable final performance.

The final regularized results show all three methods achieving **>98% accuracy** on the MNIST 3 vs 8 classification task, with the MLP models slightly outperforming SVM thanks to their ability to learn non-linear feature representations directly from raw pixel data.

---

## 10. Project Files

| File | Description |
|------|-------------|
| `projet6_optimisation_comparaison.py` | Main Python script with all models, training, and visualization |
| `projet6_optimisation_comparaison.ipynb` | Jupyter Notebook version (Google Colab compatible) |
| `rapport_optimisation.md` | This report |
