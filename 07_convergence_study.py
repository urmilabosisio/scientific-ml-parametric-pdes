# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 10:54:10 2026

@author: 15EG3019NL
"""


from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

data = np.load(RESULTS / "pde_dataset.npz")

mu_train = data["mu_train"]
u_train = data["u_train"]

mu_val = data["mu_val"]
u_val = data["u_val"]

# Validation only: keep the test set untouched.
degrees = range(1, 9)
sample_sizes = [50, 100, 200, 400]

records = []

for n_samples in sample_sizes:

    for degree in degrees:

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

        model.fit(
            mu_train[:n_samples],
            u_train[:n_samples]
        )

        prediction = model.predict(mu_val)

        errors = (
            np.linalg.norm(
                prediction - u_val,
                axis=1
            )
            /
            np.linalg.norm(
                u_val,
                axis=1
            )
        )

        records.append({
            "Samples": n_samples,
            "Degree": degree,
            "Mean L2 Error": errors.mean(),
            "Max L2 Error": errors.max()
        })

results = pd.DataFrame(records)

print("\nCONVERGENCE STUDY")
print(results.to_string(index=False))

results.to_csv(
    RESULTS / "convergence_study.csv",
    index=False
)

plt.figure(figsize=(9, 6))

for n_samples in sample_sizes:

    subset = results[
        results["Samples"] == n_samples
    ]

    plt.semilogy(
        subset["Degree"],
        subset["Mean L2 Error"],
        marker="o",
        label=f"{n_samples} samples"
    )

plt.xlabel("Polynomial degree")
plt.ylabel("Mean relative L2 error")
plt.title("Polynomial Surrogate Convergence")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTS / "convergence_study.png",
    dpi=300
)

plt.show()

print("\nConvergence study completed!")
