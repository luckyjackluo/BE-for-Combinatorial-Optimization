# Differentiable Extensions with Rounding Guarantees for Combinatorial Optimization over Permutations

This repository contains the implementation of **Birkhoff Extension (BE)**, a novel method for solving combinatorial optimization problems over permutations. BE provides a continuous, almost-everywhere differentiable extension of permutation functions to doubly stochastic matrices with theoretical guarantees.

## Abstract

Continuously extending combinatorial optimization objectives is a powerful technique commonly applied to the optimization of set functions. However, few such methods exist for extending functions on permutations, despite the fact that many combinatorial optimization problems, such as the quadratic assignment problem (QAP) and the traveling salesperson problem (TSP), are inherently optimization over permutations.

We present **Birkhoff Extension (BE)**, an almost-everywhere differentiable continuous polytime-computable extension of any real-valued function on permutations to doubly stochastic matrices. Key to this construction is our introduction of a continuous variant of the well-known Birkhoff decomposition.

## Key Features

- **Rounding Guarantee**: Any solution to the extension can be efficiently rounded to a permutation without increasing the function value
- **Differentiability**: Almost-everywhere differentiable objective function over doubly stochastic matrices
- **Gradient-based Optimization**: Amenable to gradient-descent based optimization and unsupervised neural combinatorial optimization
- **Local Improvement**: Can be combined with existing optimization approaches to offer local improvements
- **Theoretical Guarantees**: Approximate solutions in the relaxed case give rise to approximate solutions in the space of permutations

## Repository Structure

```
├── qap/                    # Quadratic Assignment Problem implementations
│   ├── solve_qap_grad_be.py    # Main BE solver for QAP
│   ├── solve_qap_2opt.py       # 2-opt local search algorithm
│   ├── solve_qap_scipy.py      # Scipy-based solver
│   ├── solve_qap_gurobi.py     # Gurobi-based solver
│   ├── train_transformer_qap.py # Neural network training
│   └── generate_qap_datasets.py # Data generation
├── tsp/                    # Traveling Salesperson Problem implementations
│   ├── Birkhoff_TSP.py         # Main BE implementation for TSP
│   ├── Birkhoff_TSP_adjacency.py # Adjacency matrix variant
│   ├── test_grad_be.py         # Gradient-based BE testing
│   ├── data_gen.py             # TSP data generation
│   └── data_convert.py         # Data format conversion
├── dfa/                    # Directed Feedback Arc Set Problem
│   ├── solve_dfasp.py          # Main DFASP solver
│   ├── test_grad_be_dfasp.py   # BE testing for DFASP
│   └── generate_data.py        # Graph data generation
├── input_data/             # Centralized data directory
│   ├── qap/               # QAP instances (real and synthetic)
│   ├── tsp/               # TSP instances and processed data
│   └── dfa/               # DFASP graph instances
└── train_be/              # Training scripts and results
```

## Installation

### Requirements

```bash
pip install torch numpy scipy networkx matplotlib tqdm
```

### Optional Dependencies

For advanced solvers:
```bash
pip install gurobi  # For Gurobi-based QAP solver
```

## Quick Start

### Quadratic Assignment Problem (QAP)

```bash
# Solve QAP using Birkhoff Extension
cd qap
python solve_qap_grad_be.py --dataset real --alg gd+k+5

# Compare with 2-opt local search
python solve_qap_2opt.py --instance tai15a.dat

# Train neural network for QAP
python train_transformer_qap.py --dataset synthetic --lr 0.01
```

### Traveling Salesperson Problem (TSP)

```bash
# Solve TSP using Birkhoff Extension
cd tsp
python test_grad_be.py --num_terms 20 --alg gd+k+5

# Generate TSP data
python data_gen.py

# Convert data formats
python data_convert.py --input tsp50-50_concorde_test.txt --num_terms 50
```

### Directed Feedback Arc Set Problem (DFASP)

```bash
# Solve DFASP using Birkhoff Extension
cd dfa
python test_grad_be_dfasp.py --n 20 --p 0.5

# Generate graph instances
python generate_data.py --ns 20 50 --ps 0.1 0.5 0.9
```

## Core Algorithms

### 1. Birkhoff Extension (BE)

The main contribution is a continuous, almost-everywhere differentiable extension of permutation functions to doubly stochastic matrices:

```python
# Core BE implementation
def cont_Birkhoff_SFE(W, k, A, B, perms, setting, device="cpu"):
    """
    Continuous Birkhoff Stochastic Function Extension
    
    Args:
        W: Score matrix for ordering permutations
        k: Number of permutations to consider
        A, B: Problem-specific matrices (e.g., distance/flow for QAP)
        perms: Set of permutation matrices
        setting: Optimization settings
    """
    # Implementation details in Birkhoff_TSP.py and solve_qap_grad_be.py
```

### 2. Iterative Proportional Fitting (IPF)

Used to maintain doubly stochastic constraints:

```python
def ipf(M, num_it, row_sum=1, col_sum=1, eps=1e-3):
    """
    Iterative Proportional Fitting to enforce doubly stochastic constraints
    """
    # Implementation in IPF.py
```

### 3. Gradient-based Optimization

Frank-Wolfe algorithm for optimization over the Birkhoff polytope:

```python
# Frank-Wolfe steps to maintain double stochasticity
# Implementation in solve_qap_grad_be.py and test_grad_be.py
```
## Data Organization

All data files are centralized in the `input_data/` directory:

- **QAP Data**: Real instances from QAPLIB and synthetic geometric instances
- **TSP Data**: Concorde-format instances and processed PyTorch tensors
- **DFASP Data**: Erdos-Rényi random graphs with various parameters

## Citation

If you use this code in your research, please cite:

```bibtex
@article{birkhoff_extension_2025,
  title={Differentiable Extensions with Rounding Guarantees for Combinatorial Optimization over Permutations},
  author={Anonymous},
  journal={NeurIPS},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions about the implementation or theoretical aspects, please open an issue on GitHub.

---

**Note**: This implementation accompanies the paper "Differentiable Extensions with Rounding Guarantees for Combinatorial Optimization over Permutations" submitted to NeurIPS 2025.
