# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 10:16:25 2026

@author: 15EG3019NL
"""


import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.linalg import solve_banded

# ==========================================
# 1. CONFIGURATION
# ==========================================

SEED = 42
N = 128

N_TRAIN = 400
N_VAL = 80
N_TEST = 80

rng = np.random.default_rng(SEED)

output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

# Interior spatial grid
h = 1.0 / (N + 1)
x = np.linspace(h, 1.0 - h, N)

# ==========================================
# 2. PARAMETRIC COEFFICIENT
# ==========================================

def diffusion(x, mu):

    mu1, mu2 = mu

    return (
        1.0
        + mu1 * x
        + mu2 * np.sin(np.pi * x)**2
    )

# ==========================================
# 3. FINITE DIFFERENCE SOLVER
# ==========================================

def solve_pde(mu):

    # Grid including boundary nodes
    x_full = np.linspace(0.0, 1.0, N + 2)

    # Midpoints of all grid intervals
    x_half = 0.5 * (
        x_full[:-1] + x_full[1:]
    )

    a_half = diffusion(x_half, mu)

    # Main diagonal
    main = (
        a_half[:-1] + a_half[1:]
    ) / h**2

    # Off-diagonals
    off = -a_half[1:-1] / h**2

    # Banded matrix representation
    ab = np.zeros((3, N))

    ab[0, 1:] = off
    ab[1, :] = main
    ab[2, :-1] = off

    # Right-hand side f(x) = 1
    rhs = np.ones(N)

    # Homogeneous Dirichlet conditions
    solution = solve_banded(
        (1, 1),
        ab,
        rhs
    )

    return solution

# ==========================================
# 4. VERIFY THE SOLVER
# ==========================================

# For mu=(0,0), a(x)=1 and:
# -u''=1, u(0)=u(1)=0
# Exact solution: u(x)=x(1-x)/2

mu_reference = np.array([0.0, 0.0])

u_numerical = solve_pde(mu_reference)
u_exact = x * (1.0 - x) / 2.0

absolute_error = np.max(
    np.abs(u_numerical - u_exact)
)

print("\nSOLVER VERIFICATION")
print("Maximum absolute error:", absolute_error)

# ==========================================
# 5. GENERATE DATASET
# ==========================================

n_samples = N_TRAIN + N_VAL + N_TEST

# Random PDE parameters
parameters = rng.uniform(
    0.0, 1.0,
    size=(n_samples, 2)
)

# Solve the PDE for every parameter
solutions = np.empty(
    (n_samples, N),
    dtype=np.float64
)

for i, mu in enumerate(parameters):
    solutions[i] = solve_pde(mu)

# ==========================================
# 6. TRAIN / VALIDATION / TEST
# ==========================================

indices = rng.permutation(n_samples)

train_idx = indices[:N_TRAIN]

val_idx = indices[
    N_TRAIN:N_TRAIN + N_VAL
]

test_idx = indices[
    N_TRAIN + N_VAL:
]

mu_train = parameters[train_idx]
u_train = solutions[train_idx]

mu_val = parameters[val_idx]
u_val = solutions[val_idx]

mu_test = parameters[test_idx]
u_test = solutions[test_idx]

# ==========================================
# 7. SAVE DATASET
# ==========================================

np.savez_compressed(
    output_dir / "pde_dataset.npz",
    x=x,
    mu_train=mu_train,
    u_train=u_train,
    mu_val=mu_val,
    u_val=u_val,
    mu_test=mu_test,
    u_test=u_test
)

print("\nDATASET GENERATED")

print("Training:", mu_train.shape, u_train.shape)
print("Validation:", mu_val.shape, u_val.shape)
print("Test:", mu_test.shape, u_test.shape)

# ==========================================
# 8. PLOT PARAMETRIC SOLUTIONS
# ==========================================

plt.figure(figsize=(9, 5))

for i in range(5):

    plt.plot(
        x,
        u_train[i],
        label=f"mu={np.round(mu_train[i], 2)}"
    )

plt.xlabel("Spatial coordinate x")
plt.ylabel("Solution u(x; mu)")

plt.title(
    "Solutions of the Parametric Elliptic PDE"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    output_dir / "parametric_solutions.png",
    dpi=300
)

plt.show()

# ==========================================
# 9. VALIDATE DATASET
# ==========================================

assert np.isfinite(solutions).all()

assert u_train.shape == (N_TRAIN, N)
assert u_val.shape == (N_VAL, N)
assert u_test.shape == (N_TEST, N)

assert np.all(solutions >= -1e-10)

assert absolute_error < 1e-8

print("\nAll checks passed!")
