# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 10:48:56 2026

@author: 15EG3019NL
"""


import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

# ==========================================
# 1. LOAD DATA
# ==========================================

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

data = np.load(RESULTS / "pde_dataset.npz")

x = data["x"]

mu_train = data["mu_train"]
u_train = data["u_train"]

mu_test = data["mu_test"]
u_test = data["u_test"]

N = len(x)
h = 1.0 / (N + 1)

x_full = np.linspace(0, 1, N + 2)
x_half = 0.5 * (x_full[:-1] + x_full[1:])

# ==========================================
# 2. TRAIN SELECTED POLYNOMIAL
# ==========================================

# Degree selected using validation data.
DEGREE = 6
ALPHA = 1e-10

model = make_pipeline(
    PolynomialFeatures(
        degree=DEGREE,
        include_bias=False
    ),
    Ridge(
        alpha=ALPHA,
        solver="svd"
    )
)

model.fit(mu_train, u_train)

# ==========================================
# 3. TEST PREDICTIONS
# ==========================================

predictions = model.predict(mu_test)

l2_errors = (
    np.linalg.norm(
        predictions - u_test,
        axis=1
    )
    /
    np.linalg.norm(
        u_test,
        axis=1
    )
)

# ==========================================
# 4. ENERGY NORM
# ==========================================

def energy_norm(v, mu):

    mu1, mu2 = mu

    coefficient = (
        1.0
        + mu1 * x_half
        + mu2 * np.sin(np.pi * x_half)**2
    )

    v_full = np.r_[0.0, v, 0.0]

    differences = np.diff(v_full)

    return np.sqrt(
        np.sum(
            coefficient * differences**2
        ) / h
    )


energy_errors = np.empty(len(mu_test))

for i, mu in enumerate(mu_test):

    error = predictions[i] - u_test[i]

    energy_errors[i] = (
        energy_norm(error, mu)
        /
        energy_norm(u_test[i], mu)
    )

# ==========================================
# 5. REPORT FINAL TEST RESULTS
# ==========================================

print("\nPOLYNOMIAL TEST RESULTS")

print("Degree:", DEGREE)

print("Mean relative L2 error:",
      l2_errors.mean())

print("Median relative L2 error:",
      np.median(l2_errors))

print("Maximum relative L2 error:",
      l2_errors.max())

print("\nPOLYNOMIAL ENERGY RESULTS")

print("Mean relative energy error:",
      energy_errors.mean())

print("Median relative energy error:",
      np.median(energy_errors))

print("Maximum relative energy error:",
      energy_errors.max())

# ==========================================
# 6. COMPARE WITH NEURAL SURROGATE
# ==========================================

# Neural surrogate test metrics from Lesson 3

nn_l2_mean = 0.0002287244999891777
nn_l2_max = 0.00044867383546033163

nn_energy_mean = 0.011227209373558127
nn_energy_max = 0.026083607774160206

comparison = pd.DataFrame({
    "Model": [
        "Neural Surrogate",
        "Polynomial Degree 6"
    ],
    "Mean L2 Error": [
        nn_l2_mean,
        l2_errors.mean()
    ],
    "Max L2 Error": [
        nn_l2_max,
        l2_errors.max()
    ],
    "Mean Energy Error": [
        nn_energy_mean,
        energy_errors.mean()
    ],
    "Max Energy Error": [
        nn_energy_max,
        energy_errors.max()
    ]
})

print("\nFINAL SURROGATE COMPARISON")
print(comparison.to_string(index=False))

comparison.to_csv(
    RESULTS / "final_surrogate_comparison.csv",
    index=False
)

# ==========================================
# 7. BENCHMARK INFERENCE
# ==========================================

# Use the same 1,000 parameters
# as the previous computational benchmark.

rng = np.random.default_rng(123)

parameters = rng.uniform(
    0.0, 1.0,
    size=(1000, 2)
)

# Warm-up
model.predict(parameters[:10])

times = []

for _ in range(20):

    start = time.perf_counter()

    output = model.predict(parameters)

    times.append(
        time.perf_counter() - start
    )

poly_time = np.median(times)

print("\nPOLYNOMIAL BENCHMARK")

print("Queries:", len(parameters))

print("Total batch time:", poly_time, "s")

print(
    "Time per query:",
    poly_time / len(parameters) * 1000,
    "ms"
)

print(
    "Speedup vs numerical solver:",
    0.023763799923472106 / poly_time
)

print(
    "Speedup vs neural batch:",
    0.0005562999285757542 / poly_time
)

# ==========================================
# 8. SAVE MODEL
# ==========================================

joblib.dump(
    model,
    RESULTS / "polynomial_surrogate.joblib"
)

print("\nPolynomial evaluation completed!")
