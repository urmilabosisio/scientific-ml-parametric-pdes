# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 10:37:30 2026

@author: 15EG3019NL
"""


import numpy as np
import torch
import matplotlib.pyplot as plt

from pathlib import Path
from torch import nn

# ==========================================
# 1. CONFIGURATION
# ==========================================

RESULTS = Path(
    r"C:\Users\15EG3019NL\Desktop\Progetto\scientific-ml-parametric-pdes\results"
)

data = np.load(RESULTS / "pde_dataset.npz")

x = data["x"]
mu_test = data["mu_test"]
u_test = data["u_test"]

# ==========================================
# 2. DEFINE NEURAL NETWORK
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

# Only load checkpoints created by you.
checkpoint = torch.load(
    RESULTS / "neural_surrogate.pt",
    map_location="cpu",
    weights_only=False
)

model = NeuralSurrogate(len(x))

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

u_mean = checkpoint["u_mean"]
u_std = checkpoint["u_std"]

# ==========================================
# 3. TEST PREDICTIONS
# ==========================================

mu_scaled = 2.0 * mu_test - 1.0

with torch.no_grad():

    input_tensor = torch.tensor(
        mu_scaled,
        dtype=torch.float32
    )

    pred_scaled = model(
        input_tensor
    ).numpy()

u_pred = pred_scaled * u_std + u_mean

# ==========================================
# 4. RELATIVE L2 ERROR
# ==========================================

l2_errors = (
    np.linalg.norm(u_pred - u_test, axis=1)
    /
    np.linalg.norm(u_test, axis=1)
)

# ==========================================
# 5. RELATIVE ENERGY ERROR
# ==========================================

N = len(x)
h = 1.0 / (N + 1)

x_full = np.linspace(0.0, 1.0, N + 2)
x_half = 0.5 * (x_full[:-1] + x_full[1:])

def energy_norm(v, mu):

    mu1, mu2 = mu

    coefficient = (
        1.0
        + mu1 * x_half
        + mu2 * np.sin(np.pi * x_half)**2
    )

    # Homogeneous boundary conditions
    v_full = np.r_[0.0, v, 0.0]

    differences = np.diff(v_full)

    return np.sqrt(
        np.sum(coefficient * differences**2) / h
    )

energy_errors = []

for i in range(len(mu_test)):

    mu = mu_test[i]

    numerator = energy_norm(
        u_pred[i] - u_test[i],
        mu
    )

    denominator = energy_norm(
        u_test[i],
        mu
    )

    energy_errors.append(
        numerator / denominator
    )

energy_errors = np.array(energy_errors)

# ==========================================
# 6. PRINT RESULTS
# ==========================================

print("\nFINAL TEST RESULTS")

print("Mean relative L2 error:",
      l2_errors.mean())

print("Median relative L2 error:",
      np.median(l2_errors))

print("Maximum relative L2 error:",
      l2_errors.max())

print("\nENERGY NORM RESULTS")

print("Mean relative energy error:",
      energy_errors.mean())

print("Median relative energy error:",
      np.median(energy_errors))

print("Maximum relative energy error:",
      energy_errors.max())

# ==========================================
# 7. VISUAL COMPARISON
# ==========================================

# Use the largest energy error case
idx = np.argmax(energy_errors)

x_plot = np.r_[0.0, x, 1.0]

reference = np.r_[0.0, u_test[idx], 0.0]
prediction = np.r_[0.0, u_pred[idx], 0.0]

plt.figure(figsize=(9, 5))

plt.plot(
    x_plot,
    reference,
    label="Numerical solver",
    linewidth=2
)

plt.plot(
    x_plot,
    prediction,
    "--",
    label="Neural surrogate",
    linewidth=2
)

plt.xlabel("x")
plt.ylabel("u(x)")

plt.title(
    "Test Solution - Largest Relative Energy Error"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTS / "test_solution_comparison.png",
    dpi=300
)

plt.show()

# ==========================================
# 8. SAVE METRICS
# ==========================================

np.savez_compressed(
    RESULTS / "test_metrics.npz",
    l2_errors=l2_errors,
    energy_errors=energy_errors
)

print("\nTest evaluation completed!")
