# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 10:45:45 2026

@author: 15EG3019NL
"""


import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from scipy.linalg import solve_banded

# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

N_QUERIES = 1000
REPEATS = 5
SEED = 123

torch.set_num_threads(1)

data = np.load(RESULTS / "pde_dataset.npz")
x = data["x"]

N = len(x)
h = 1.0 / (N + 1)

rng = np.random.default_rng(SEED)

parameters = rng.uniform(
    0, 1, size=(N_QUERIES, 2)
)

# ==========================================
# 2. NUMERICAL SOLVER
# ==========================================

x_full = np.linspace(0, 1, N + 2)
x_half = (x_full[:-1] + x_full[1:]) / 2

rhs = np.ones(N)

def solve_pde(mu):

    mu1, mu2 = mu

    a = (
        1.0
        + mu1 * x_half
        + mu2 * np.sin(np.pi * x_half)**2
    )

    main = (a[:-1] + a[1:]) / h**2
    off = -a[1:-1] / h**2

    ab = np.zeros((3, N))
    ab[0, 1:] = off
    ab[1, :] = main
    ab[2, :-1] = off

    return solve_banded(
        (1, 1), ab, rhs,
        check_finite=False
    )

# ==========================================
# 3. NEURAL SURROGATE
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

model = NeuralSurrogate(N)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

mean = checkpoint["u_mean"]
std = checkpoint["u_std"]

# ==========================================
# 4. PREDICTION FUNCTIONS
# ==========================================

def numerical_predictions():

    return np.array([
        solve_pde(mu)
        for mu in parameters
    ])


def neural_predictions_single():

    predictions = np.empty(
        (N_QUERIES, N)
    )

    with torch.inference_mode():

        for i, mu in enumerate(parameters):

            inputs = torch.tensor(
                2 * mu - 1,
                dtype=torch.float32
            ).unsqueeze(0)

            output = model(inputs).numpy()

            predictions[i] = (
                output[0] * std + mean
            )

    return predictions


def neural_predictions_batch():

    with torch.inference_mode():

        inputs = torch.tensor(
            2 * parameters - 1,
            dtype=torch.float32
        )

        outputs = model(inputs).numpy()

    return outputs * std + mean

# ==========================================
# 5. WARM-UP
# ==========================================

solve_pde(parameters[0])

with torch.inference_mode():
    model(torch.zeros((1, 2)))
    model(torch.zeros((N_QUERIES, 2)))

# ==========================================
# 6. BENCHMARK
# ==========================================

def benchmark(function):

    times = []

    result = None

    for _ in range(REPEATS):

        start = time.perf_counter()

        result = function()

        elapsed = time.perf_counter() - start

        times.append(elapsed)

    return np.median(times), result


solver_time, u_solver = benchmark(
    numerical_predictions
)

single_time, u_single = benchmark(
    neural_predictions_single
)

batch_time, u_batch = benchmark(
    neural_predictions_batch
)

# ==========================================
# 7. ACCURACY CHECK
# ==========================================

relative_errors = (
    np.linalg.norm(
        u_batch - u_solver, axis=1
    )
    /
    np.linalg.norm(
        u_solver, axis=1
    )
)

assert np.allclose(
    u_single, u_batch, rtol=1e-5, atol=1e-6
)

# ==========================================
# 8. RESULTS
# ==========================================

print("\nCOMPUTATIONAL BENCHMARK")
print("Number of queries:", N_QUERIES)

print("\nTOTAL EXECUTION TIME")

print("Numerical solver:", solver_time, "s")
print("Neural single:", single_time, "s")
print("Neural batch:", batch_time, "s")

print("\nAVERAGE TIME PER QUERY")

print("Numerical solver:",
      solver_time / N_QUERIES * 1000, "ms")

print("Neural single:",
      single_time / N_QUERIES * 1000, "ms")

print("Neural batch throughput equivalent:",
      batch_time / N_QUERIES * 1000, "ms")

print("\nSPEEDUP")

print("Single inference:",
      solver_time / single_time)

print("Batch inference:",
      solver_time / batch_time)

print("\nACCURACY ON BENCHMARK PARAMETERS")

print("Mean relative L2 error:",
      relative_errors.mean())

print("Maximum relative L2 error:",
      relative_errors.max())

print("\nBenchmark completed!")
