# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 10:56:05 2026

@author: 15EG3019NL
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.integrate import quad
from scipy.linalg import solve_banded
from numpy.polynomial.legendre import leggauss

# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

N_VALUES = [16, 32, 64, 128, 256, 512]

PARAMETERS = [
    (0.25, 0.50),
    (0.75, 0.25),
    (1.00, 1.00)
]

# ==========================================
# 2. DIFFUSION COEFFICIENT
# ==========================================

def a(x, mu):
    mu1, mu2 = mu

    return (
        1.0
        + mu1 * x
        + mu2 * np.sin(np.pi * x)**2
    )

# ==========================================
# 3. EXACT SOLUTION
# ==========================================

def exact_constant(mu):

    numerator = quad(
        lambda t: t / a(t, mu),
        0.0, 1.0,
        epsabs=1e-12,
        epsrel=1e-12
    )[0]

    denominator = quad(
        lambda t: 1.0 / a(t, mu),
        0.0, 1.0,
        epsabs=1e-12,
        epsrel=1e-12
    )[0]

    return numerator / denominator


def exact_solution(x, mu, C=None):

    if C is None:
        C = exact_constant(mu)

    return np.array([
        quad(
            lambda t: (C - t) / a(t, mu),
            0.0, float(xi),
            epsabs=1e-12,
            epsrel=1e-12
        )[0]
        for xi in np.atleast_1d(x)
    ])


def exact_derivative(x, mu, C):

    return (C - x) / a(x, mu)

# ==========================================
# 4. FINITE DIFFERENCE SOLVER
# ==========================================

def solve_pde(N, mu):

    h = 1.0 / (N + 1)

    x = np.linspace(0.0, 1.0, N + 2)
    x_half = 0.5 * (x[:-1] + x[1:])

    ah = a(x_half, mu)

    main = (ah[:-1] + ah[1:]) / h**2
    off = -ah[1:-1] / h**2

    ab = np.zeros((3, N))

    ab[0, 1:] = off
    ab[1, :] = main
    ab[2, :-1] = off

    u_inner = solve_banded(
        (1, 1),
        ab,
        np.ones(N),
        check_finite=False
    )

    u_full = np.r_[0.0, u_inner, 0.0]

    return x, u_full

# ==========================================
# 5. ERROR NORMS
# ==========================================

def relative_l2(u_num, u_exact):

    # Discrete, h-weighted nodal L2 norm.
    return (
        np.linalg.norm(u_num - u_exact)
        /
        np.linalg.norm(u_exact)
    )


def relative_energy(x, u_num, mu, C):

    # Piecewise-linear interpolant of nodal values.
    h = x[1] - x[0]

    slopes = np.diff(u_num) / h

    # Gaussian quadrature on each cell.
    nodes, weights = leggauss(6)

    left = x[:-1, None]

    quadrature_points = (
        left + 0.5 * h * (nodes[None, :] + 1.0)
    )

    ux_exact = exact_derivative(
        quadrature_points, mu, C
    )

    coeff = a(quadrature_points, mu)

    squared_error = (
        coeff * (slopes[:, None] - ux_exact)**2
    )

    squared_reference = coeff * ux_exact**2

    numerator = np.sum(
        squared_error * weights[None, :]
    ) * h / 2.0

    denominator = np.sum(
        squared_reference * weights[None, :]
    ) * h / 2.0

    return np.sqrt(numerator / denominator)

# ==========================================
# 6. DISCRETIZATION STUDY
# ==========================================

records = []

for mu in PARAMETERS:

    C = exact_constant(mu)

    for N in N_VALUES:

        x, u_num = solve_pde(N, mu)

        u_exact = exact_solution(x, mu, C)

        h = 1.0 / (N + 1)

        e_l2 = relative_l2(u_num, u_exact)

        e_energy = relative_energy(
            x, u_num, mu, C
        )

        records.append({
            "mu1": mu[0],
            "mu2": mu[1],
            "N": N,
            "h": h,
            "L2 Error": e_l2,
            "Energy Error": e_energy
        })

results = pd.DataFrame(records)

print("\nDISCRETIZATION STUDY")
print(results.to_string(index=False))

results.to_csv(
    RESULTS / "discretization_study.csv",
    index=False
)

# ==========================================
# 7. ESTIMATED CONVERGENCE ORDERS
# ==========================================

print("\nESTIMATED CONVERGENCE ORDERS")

for mu in PARAMETERS:

    subset = results[
        (results["mu1"] == mu[0])
        & (results["mu2"] == mu[1])
    ].sort_values("N")

    h = subset["h"].to_numpy()

    e_l2 = subset["L2 Error"].to_numpy()
    e_energy = subset["Energy Error"].to_numpy()

    p_l2 = (
        np.log(e_l2[:-1] / e_l2[1:])
        /
        np.log(h[:-1] / h[1:])
    )

    p_energy = (
        np.log(e_energy[:-1] / e_energy[1:])
        /
        np.log(h[:-1] / h[1:])
    )

    print(f"\nmu = {mu}")

    for i in range(len(p_l2)):

        print(
            f"N={N_VALUES[i+1]:3d} | "
            f"L2 order={p_l2[i]:.4f} | "
            f"Energy order={p_energy[i]:.4f}"
        )

# ==========================================
# 8. CONVERGENCE PLOTS
# ==========================================

for error_column, filename, ylabel in [
    (
        "L2 Error",
        "discretization_l2.png",
        "Relative discrete L2 error"
    ),
    (
        "Energy Error",
        "discretization_energy.png",
        "Relative energy error"
    )
]:

    plt.figure(figsize=(8, 5))

    for mu in PARAMETERS:

        subset = results[
            (results["mu1"] == mu[0])
            & (results["mu2"] == mu[1])
        ].sort_values("h")

        plt.loglog(
            subset["h"],
            subset[error_column],
            marker="o",
            label=f"mu={mu}"
        )

    plt.xlabel("Mesh size h")
    plt.ylabel(ylabel)
    plt.title("Spatial Discretization Convergence")

    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS / filename,
        dpi=300
    )

    plt.show()

# ==========================================
# 9. N=128: COMPARE WITH SURROGATE
# ==========================================

print("\nDISCRETIZATION ERROR AT N=128")

dataset_path = RESULTS / "pde_dataset.npz"

if dataset_path.exists():

    dataset = np.load(dataset_path)

    mu_test = dataset["mu_test"]
    u_test = dataset["u_test"]

    x_inner = dataset["x"]

    exact_test = np.empty_like(u_test)

    for i, mu in enumerate(mu_test):

        C = exact_constant(mu)

        exact_test[i] = exact_solution(
            x_inner, mu, C
        )

    disc_errors = np.linalg.norm(
        u_test - exact_test,
        axis=1
    ) / np.linalg.norm(
        exact_test,
        axis=1
    )

    print(
        "Mean relative discretization L2 error:",
        disc_errors.mean()
    )

    print(
        "Max relative discretization L2 error:",
        disc_errors.max()
    )

    # Optional comparison with polynomial model.
    poly_path = RESULTS / "polynomial_surrogate.joblib"

    if poly_path.exists():

        import joblib

        poly_model = joblib.load(poly_path)

        poly_predictions = poly_model.predict(mu_test)

        total_errors = np.linalg.norm(
            poly_predictions - exact_test,
            axis=1
        ) / np.linalg.norm(
            exact_test,
            axis=1
        )

        surrogate_errors = np.linalg.norm(
            poly_predictions - u_test,
            axis=1
        ) / np.linalg.norm(
            u_test,
            axis=1
        )

        print(
            "Mean polynomial surrogate L2 error:",
            surrogate_errors.mean()
        )

        print(
            "Mean polynomial total L2 error:",
            total_errors.mean()
        )

print("\nDiscretization study completed!")
