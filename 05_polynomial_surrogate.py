# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 10:47:40 2026

@author: 15EG3019NL
"""


import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

# ==========================================
# 1. LOAD DATA
# ==========================================

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

data = np.load(RESULTS / "pde_dataset.npz")

mu_train = data["mu_train"]
u_train = data["u_train"]

mu_val = data["mu_val"]
u_val = data["u_val"]

# ==========================================
# 2. RELATIVE L2 ERROR
# ==========================================

def relative_l2(pred, reference):
    return (
        np.linalg.norm(pred - reference, axis=1)
        / np.linalg.norm(reference, axis=1)
    )

# ==========================================
# 3. POLYNOMIAL SURROGATE
# ==========================================

results = []
models = {}

for degree in [1, 2, 3, 4, 5, 6]:

    model = make_pipeline(
        PolynomialFeatures(
            degree=degree,
            include_bias=False
        ),
        Ridge(
            alpha=1e-10,
            solver="svd"
        )
    )

    model.fit(mu_train, u_train)

    prediction = model.predict(mu_val)

    errors = relative_l2(
        prediction, u_val
    )

    results.append({
        "Degree": degree,
        "Mean L2 Error": errors.mean(),
        "Median L2 Error": np.median(errors),
        "Maximum L2 Error": errors.max()
    })

    models[degree] = model

comparison = pd.DataFrame(results)

print("\nPOLYNOMIAL SURROGATE RESULTS")
print(comparison.to_string(index=False))

best_idx = comparison["Mean L2 Error"].idxmin()

best_degree = int(
    comparison.loc[best_idx, "Degree"]
)

print("\nBEST POLYNOMIAL DEGREE:", best_degree)

print(
    "BEST VALIDATION MEAN L2 ERROR:",
    comparison.loc[best_idx, "Mean L2 Error"]
)

comparison.to_csv(
    RESULTS / "polynomial_comparison.csv",
    index=False
)

print("\nPolynomial comparison completed!")
