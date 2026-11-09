# Scientific Machine Learning for Parametric PDEs

### Neural Networks vs Polynomial Surrogate Models

An experimental study of **neural-network and polynomial surrogate models for parametric elliptic partial differential equations**, with a focus on approximation accuracy, energy-norm errors, convergence, and computational efficiency.

## Project Overview

This project investigates whether a neural network is necessary to efficiently approximate the solution operator of a smooth, low-dimensional parametric PDE.

Three approaches are compared:

1. A classical finite-difference solver.
2. A fully connected neural network implemented in PyTorch.
3. A multivariate polynomial surrogate constructed using regularized least squares.

The results demonstrate that, for the considered problem, a low-degree polynomial surrogate substantially outperforms the neural network in accuracy while achieving competitive inference performance.

## Mathematical Problem

Consider the boundary-value problem

\[
-\frac{d}{dx}\left(a(x;\mu)\frac{du}{dx}\right)=1,
\quad x\in(0,1),
\]

with homogeneous Dirichlet boundary conditions

\[
u(0;\mu)=u(1;\mu)=0,
\]

and diffusion coefficient

\[
a(x;\mu)=1+\mu_1x+\mu_2\sin^2(\pi x),
\quad \mu\in[0,1]^2.
\]

The coefficient is uniformly elliptic. The goal is to approximate the parameter-to-solution map

\[
\mu\mapsto u_h(\mu).
\]

## Numerical Method

The PDE is discretized using a conservative finite-difference scheme with 128 interior nodes. The resulting symmetric positive definite tridiagonal system is solved using SciPy's banded linear solver.

The dataset consists of:

| Dataset | Samples |
|---|---:|
| Training | 400 |
| Validation | 80 |
| Test | 80 |

The test set is kept separate from model selection.

## Neural Network Surrogate

A fully connected multilayer perceptron is implemented in PyTorch with architecture:

`2 → 64 → 64 → 128`

The model uses hyperbolic tangent activations, Adam optimization, output standardization, and early stopping based on validation loss.

## Polynomial Surrogate

A total-degree polynomial approximation is constructed using multivariate monomials and Ridge regression.

Polynomial degrees 1–6 are compared on the validation set, and degree 6 is selected for the independent test evaluation.

An additional convergence study investigates degrees 1–8 using 50, 100, 200, and 400 training samples.

## Test Results

| Metric | Neural Network | Degree-6 Polynomial |
|---|---:|---:|
| Mean relative L2 error | 2.287 × 10⁻⁴ | **1.716 × 10⁻⁶** |
| Maximum relative L2 error | 4.487 × 10⁻⁴ | **6.751 × 10⁻⁶** |
| Mean relative energy error | 1.123 × 10⁻² | **3.685 × 10⁻⁶** |
| Maximum relative energy error | 2.608 × 10⁻² | **1.480 × 10⁻⁵** |

The polynomial surrogate achieves approximately **133 times lower mean relative L2 error** and **3,046 times lower mean relative energy error** than the neural network.

## Computational Benchmark

CPU inference performance is measured using identical parameter batches and controlled numerical-library thread pools.

Speedup relative to the finite-difference solver:

| Batch size | Neural Network | Polynomial |
|---|---:|---:|
| 1 | 0.56× | 0.33× |
| 10 | 4.67× | 1.43× |
| 100 | 15.38× | 10.49× |
| 1,000 | 30.11× | 72.45× |
| 5,000 | 24.38× | 31.06× |

The measured degree-6 polynomial fitting time is approximately 1.57 ms, excluding dataset generation.

The original neural-network training time was not recorded. Therefore, a complete offline-cost or break-even comparison is not reported.

## Convergence Analysis

The polynomial surrogate exhibits rapidly decreasing validation error with increasing degree.

With 400 training samples, the mean relative L2 error decreases from approximately 1.32 × 10⁻² at degree 1 to 6.42 × 10⁻⁸ at degree 8.

A separate spatial-discretization study confirms the expected convergence orders:

- Second-order convergence in the discrete nodal L2 norm.
- First-order convergence in the energy norm for the piecewise-linear interpolant.

At 128 interior nodes, the mean relative discretization error is approximately 6.71 × 10⁻⁶, compared with 1.72 × 10⁻⁶ for the degree-6 polynomial surrogate relative to the discrete solution.

## Main Findings

The experiments suggest that classical approximation methods can be particularly effective for smooth, low-dimensional parametric PDEs.

Although neural networks provide efficient batched predictions, their additional modeling complexity does not necessarily result in superior accuracy.

The study illustrates the importance of combining numerical analysis, approximation theory, and machine learning when designing surrogate models.

## Limitations

The study is restricted to a one-dimensional elliptic problem with two parameters, a fixed spatial resolution for surrogate training, and one neural-network architecture.

The benchmark results depend on the hardware and numerical-library configuration. The study does not establish universal performance rankings or theoretical convergence rates for the fitted surrogate models.

## Technologies

Python · NumPy · SciPy · PyTorch · scikit-learn · pandas · Matplotlib · Numerical PDEs · Scientific Machine Learning · Polynomial Approximation

## Author

**Urmila Bosisio**

PhD Candidate in Applied Mathematics — University of Milan

Research interests: Constructive Approximation, Adaptive Numerical Methods for PDEs, Scientific Computing, Data Science, and Machine Learning.
