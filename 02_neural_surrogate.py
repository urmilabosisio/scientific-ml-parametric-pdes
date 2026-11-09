# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 10:19:23 2026

@author: 15EG3019NL
"""


# ==========================================
# SCIENTIFIC MACHINE LEARNING
# LESSON 2: NEURAL SURROGATE
# ==========================================

import copy
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# 1. CONFIGURATION
# ==========================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.set_num_threads(2)

OUTPUT_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = OUTPUT_DIR / "pde_dataset.npz"
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cpu")

EPOCHS = 1200
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
PATIENCE = 120

# ==========================================
# 2. LOAD DATASET
# ==========================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Dataset not found. Run 01_pde_solver.py first."
    )

data = np.load(DATA_PATH)

x = data["x"]

mu_train = data["mu_train"]
u_train = data["u_train"]

mu_val = data["mu_val"]
u_val = data["u_val"]

print("\nDATASET LOADED")
print("Training:", mu_train.shape, u_train.shape)
print("Validation:", mu_val.shape, u_val.shape)

# ==========================================
# 3. NORMALIZATION
# ==========================================

# Parameters already belong to [0,1]^2.
# Center them in [-1,1]^2.

mu_train_scaled = 2.0 * mu_train - 1.0
mu_val_scaled = 2.0 * mu_val - 1.0

# Normalize solutions using training data only.

u_mean = u_train.mean(axis=0)
u_std = u_train.std(axis=0)

u_std = np.maximum(u_std, 1e-8)

u_train_scaled = (u_train - u_mean) / u_std
u_val_scaled = (u_val - u_mean) / u_std

# ==========================================
# 4. CREATE PYTORCH DATASETS
# ==========================================

def to_tensor(array):
    return torch.tensor(array, dtype=torch.float32)

train_dataset = TensorDataset(
    to_tensor(mu_train_scaled),
    to_tensor(u_train_scaled)
)

val_inputs = to_tensor(mu_val_scaled)
val_targets = to_tensor(u_val_scaled)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# ==========================================
# 5. NEURAL NETWORK
# ==========================================

class NeuralSurrogate(nn.Module):

    def __init__(self, n_outputs):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),

            nn.Linear(64, 64),
            nn.Tanh(),

            nn.Linear(64, n_outputs)
        )

    def forward(self, mu):
        return self.network(mu)


model = NeuralSurrogate(
    n_outputs=len(x)
).to(DEVICE)

print("\nMODEL ARCHITECTURE")
print(model)

# ==========================================
# 6. TRAINING SETUP
# ==========================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

loss_function = nn.MSELoss()

best_val_loss = float("inf")
best_state = None

train_losses = []
val_losses = []

epochs_without_improvement = 0

# ==========================================
# 7. TRAINING LOOP
# ==========================================

for epoch in range(EPOCHS):

    model.train()

    epoch_loss = 0.0

    for inputs, targets in train_loader:

        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()

        predictions = model(inputs)

        loss = loss_function(
            predictions,
            targets
        )

        loss.backward()
        optimizer.step()

        epoch_loss += (
            loss.item() * inputs.size(0)
        )

    epoch_loss /= len(train_dataset)

    # Validation
    model.eval()

    with torch.no_grad():

        val_predictions = model(
            val_inputs.to(DEVICE)
        )

        val_loss = loss_function(
            val_predictions,
            val_targets.to(DEVICE)
        ).item()

    train_losses.append(epoch_loss)
    val_losses.append(val_loss)

    # Early stopping
    if val_loss < best_val_loss - 1e-6:

        best_val_loss = val_loss

        best_state = copy.deepcopy(
            model.state_dict()
        )

        best_epoch = epoch + 1
        epochs_without_improvement = 0

    else:
        epochs_without_improvement += 1

    if (epoch + 1) % 100 == 0:

        print(
            f"Epoch {epoch + 1:4d} | "
            f"Train MSE: {epoch_loss:.6f} | "
            f"Validation MSE: {val_loss:.6f}"
        )

    if epochs_without_improvement >= PATIENCE:

        print(
            f"\nEarly stopping at epoch {epoch + 1}"
        )
        break

# Restore the best validation model
model.load_state_dict(best_state)
model.eval()

print("\nTRAINING COMPLETED")
print("Best epoch:", best_epoch)
print("Best validation MSE:", best_val_loss)

# ==========================================
# 8. VALIDATION PREDICTIONS
# ==========================================

with torch.no_grad():

    predictions_scaled = model(
        val_inputs.to(DEVICE)
    ).cpu().numpy()

# Return to physical scale
predictions = (
    predictions_scaled * u_std + u_mean
)

# ==========================================
# 9. RELATIVE L2 ERRORS
# ==========================================

# On a uniform grid, the common
# sqrt(h) factor cancels in the ratio.

relative_errors = (
    np.linalg.norm(
        predictions - u_val,
        axis=1
    )
    /
    np.linalg.norm(
        u_val,
        axis=1
    )
)

print("\nVALIDATION RESULTS")

print(
    "Mean relative L2 error:",
    relative_errors.mean()
)

print(
    "Median relative L2 error:",
    np.median(relative_errors)
)

print(
    "Maximum relative L2 error:",
    relative_errors.max()
)

# ==========================================
# 10. TRAINING CURVES
# ==========================================

plt.figure(figsize=(9, 5))

plt.semilogy(
    train_losses,
    label="Training loss"
)

plt.semilogy(
    val_losses,
    label="Validation loss"
)

plt.axvline(
    best_epoch - 1,
    linestyle="--",
    label="Best epoch"
)

plt.xlabel("Epoch")
plt.ylabel("Normalized MSE")

plt.title("Neural Surrogate - Training History")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "training_history.png",
    dpi=300
)

plt.show()

# ==========================================
# 11. COMPARE PDE SOLUTIONS
# ==========================================

# Best and worst validation samples
best_idx = np.argmin(relative_errors)
worst_idx = np.argmax(relative_errors)

for idx, label in [
    (best_idx, "best"),
    (worst_idx, "worst")
]:

    # Add exact zero boundary values
    x_full = np.r_[0.0, x, 1.0]
    u_reference = np.r_[0.0, u_val[idx], 0.0]
    u_predicted = np.r_[0.0, predictions[idx], 0.0]

    plt.figure(figsize=(9, 5))

    plt.plot(
        x_full,
        u_reference,
        label="Numerical solver",
        linewidth=2
    )

    plt.plot(
        x_full,
        u_predicted,
        "--",
        label="Neural surrogate",
        linewidth=2
    )

    plt.xlabel("x")
    plt.ylabel("u(x)")

    plt.title(
        f"{label.capitalize()} validation example "
        f"(relative L2 error: "
        f"{relative_errors[idx]:.4%})"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{label}_validation_solution.png",
        dpi=300
    )

    plt.show()

# ==========================================
# 12. SAVE MODEL
# ==========================================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "n_outputs": len(x),
        "u_mean": u_mean,
        "u_std": u_std,
        "best_epoch": best_epoch
    },
    OUTPUT_DIR / "neural_surrogate.pt"
)

print("\nModel saved successfully!")
