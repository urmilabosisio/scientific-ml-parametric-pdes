# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 10:59:52 2026

@author: 15EG3019NL
"""


from pathlib import Path
from time import perf_counter
from contextlib import contextmanager

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import torch

from torch import nn
from scipy.linalg import solve_banded
from threadpoolctl import threadpool_limits

# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

SEED = 123
BATCH_SIZES = [1, 10, 100, 1000, 5000]
REPEATS = 20

torch.set_num_threads(1)

data = np.load(RESULTS / "pde_dataset.npz")

x = data["x"]
mu_train = data["mu_train"]
u_train = data["u_train"]

N = len(x)
h = 1.0 / (N + 1)

rng = np.random.default_rng(SEED)
parameters = rng.uniform(
    0.0, 1.0,
    size=(max(BATCH_SIZES), 2)
)

# ==========================================
# 2. NUMERICAL SOLVER
# ==========================================

x_full = np.linspace(0, 1, N + 2)
x_half = 0.5 * (x_full[:-1] + x_full[1:])

sin_half_squared = np.sin(
    np.pi * x_half
)**2

rhs = np.ones(N)

def solve_pde(mu):

    mu1, mu2 = mu

    a = (
        1.0
        + mu1 * x_half
        + mu2 * sin_half_squared
    )

    main = (a[:-1] + a[1:]) / h**2
    off = -a[1:-1] / h**2

    ab = np.zeros((3, N))

    ab[0, 1:] = off
    ab[1, :] = main
    ab[2, :-1] = off

    return solve_banded(
        (1, 1),
        ab,
        rhs,
        check_finite=False
    )


def numerical_batch(mu):
    return np.array([
        solve_pde(p)
        for p in mu
    ])

# ==========================================
# 3. LOAD NEURAL SURROGATE
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


checkpoint = torch.load(
    RESULTS / "neural_surrogate.pt",
    map_location="cpu",
    weights_only=False
)

neural_model = NeuralSurrogate(N)

neural_model.load_state_dict(
    checkpoint["model_state_dict"]
)

neural_model.eval()

u_mean = np.asarray(checkpoint["u_mean"])
u_std = np.asarray(checkpoint["u_std"])


def neural_batch(mu):

    inputs = torch.as_tensor(
        2.0 * mu - 1.0,
        dtype=torch.float32
    )

    with torch.inference_mode():
        output = neural_model(inputs).numpy()

    return output * u_std + u_mean

# ==========================================
# 4. LOAD POLYNOMIAL SURROGATE
# ==========================================

polynomial_model = joblib.load(
    RESULTS / "polynomial_surrogate.joblib"
)


def polynomial_batch(mu):
    return polynomial_model.predict(mu)

# ==========================================
# 5. TIMING FUNCTION
# ==========================================

def measure_time(func, inputs):

    # Warm-up
    func(inputs)

    times = []

    for _ in range(REPEATS):

        start = perf_counter()
        func(inputs)
        elapsed = perf_counter() - start

        times.append(elapsed)

    return float(np.median(times))


def relative_l2(pred, ref):

    return (
        np.linalg.norm(pred - ref, axis=1)
        / np.linalg.norm(ref, axis=1)
    )

# ==========================================
# 6. CONTROLLED BENCHMARK
# ==========================================

records = []

# Restrict BLAS/OpenMP thread pools.
with threadpool_limits(limits=1):

    for batch_size in BATCH_SIZES:

        mu = parameters[:batch_size]

        reference = numerical_batch(mu)

        methods = {
            "Numerical Solver": numerical_batch,
            "Neural Surrogate": neural_batch,
            "Polynomial Surrogate": polynomial_batch
        }

        for name, function in methods.items():

            elapsed = measure_time(function, mu)

            prediction = function(mu)

            if name == "Numerical Solver":
                error_mean = 0.0
                error_max = 0.0
            else:
                errors = relative_l2(
                    prediction, reference
                )
                error_mean = float(errors.mean())
                error_max = float(errors.max())

            records.append({
                "Method": name,
                "Batch Size": batch_size,
                "Total Time (s)": elapsed,
                "Time per Query (ms)":
                    elapsed / batch_size * 1000,
                "Mean L2 Error": error_mean,
                "Max L2 Error": error_max
            })

            print(
                f"{name:22s} | "
                f"Batch={batch_size:5d} | "
                f"Time={elapsed:.8f} s"
            )

# ==========================================
# 7. SAVE RESULTS
# ==========================================

benchmark = pd.DataFrame(records)

benchmark.to_csv(
    RESULTS / "final_benchmark.csv",
    index=False
)

print("\nFINAL BENCHMARK")
print(
    benchmark.to_string(
        index=False,
        float_format=lambda v: f"{v:.8g}"
    )
)

# ==========================================
# 8. SPEEDUP SUMMARY
# ==========================================

print("\nSPEEDUP SUMMARY")

for batch_size in BATCH_SIZES:

    subset = benchmark[
        benchmark["Batch Size"] == batch_size
    ]

    def get_time(method):
        return float(
            subset.loc[
                subset["Method"] == method,
                "Total Time (s)"
            ].iloc[0]
        )

    solver_t = get_time("Numerical Solver")
    neural_t = get_time("Neural Surrogate")
    poly_t = get_time("Polynomial Surrogate")

    print(
        f"Batch {batch_size:5d}: "
        f"NN={solver_t/neural_t:.2f}x, "
        f"Poly={solver_t/poly_t:.2f}x"
    )

# ==========================================
# 9. PLOT BATCH PERFORMANCE
# ==========================================

plt.figure(figsize=(9, 6))

for name in [
    "Numerical Solver",
    "Neural Surrogate",
    "Polynomial Surrogate"
]:

    subset = benchmark[
        benchmark["Method"] == name
    ].sort_values("Batch Size")

    plt.loglog(
        subset["Batch Size"],
        subset["Total Time (s)"],
        marker="o",
        label=name
    )

plt.xlabel("Number of queries")
plt.ylabel("Total execution time (s)")
plt.title("Controlled CPU Benchmark")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    RESULTS / "final_benchmark.png",
    dpi=300
)

plt.show()

# ==========================================
# 10. POLYNOMIAL OFFLINE COST
# ==========================================

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

training_times = []

with threadpool_limits(limits=1):

    for _ in range(5):

        model = make_pipeline(
            PolynomialFeatures(
                degree=6,
                include_bias=False
            ),
            Ridge(
                alpha=1e-10,
                solver="svd"
            )
        )

        start = perf_counter()
        model.fit(mu_train, u_train)
        training_times.append(
            perf_counter() - start
        )

poly_training_time = float(
    np.median(training_times)
)

# ==========================================
# 11. OFFLINE COST
# ==========================================

print("\nOFFLINE COST")

print(
    "Polynomial training time:",
    poly_training_time, "s"
)

print(
    "Neural training time:",
    "Not recorded during the original run"
)

print(
    "Dataset generation time:",
    "Not recorded during the original run"
)

print(
    "\nNote: offline costs are not directly "
    "comparable until all are measured."
)

print("\nFinal benchmark completed!")
