# 2-Opt Algorithm for Quadratic Assignment Problem (QAP)

This implementation provides a 2-opt local search algorithm for solving the Quadratic Assignment Problem, based on the approach described in:

> G. A. Croes. A method for solving traveling-salesman problems. Operations Research, 6(6):791-812, 1958.

While Croes' original work focused on the Traveling Salesman Problem, the 2-opt principle has been successfully adapted for the QAP.

## Problem Description

The Quadratic Assignment Problem (QAP) seeks to assign `n` facilities to `n` locations such that the total cost is minimized. Given:
- **Distance matrix** `D`: `D[i,j]` = distance between locations `i` and `j`
- **Flow matrix** `F`: `F[i,j]` = flow between facilities `i` and `j`

The objective is to find a permutation `π` that minimizes:
```
∑∑ D[i,j] × F[π(i), π(j)]
i j
```

## Algorithm Overview

The 2-opt algorithm for QAP works by:

1. **Initialize**: Start with an initial assignment (random or heuristic)
2. **Local Search**: For each pair of positions (i,j), evaluate swapping the facilities
3. **Improve**: If a swap reduces the objective value, accept it
4. **Iterate**: Repeat until no improving swaps are found (local optimum)
5. **Restart**: Use multiple random restarts to escape local optima

## Files

- **`qap_2opt.py`**: Main implementation of the 2-opt algorithm
- **`test_2opt_vs_scipy.py`**: Comparison script with scipy's quadratic assignment solver
- **`demo_2opt.py`**: Demonstration script showing the algorithm in action

## Usage

### Basic Usage

```bash
# Solve a specific QAP instance
python qap_2opt.py --instance qap/qap/prob/tai10a.dat

# Solve with custom parameters
python qap_2opt.py --instance tai10a.dat --max_iterations 1000 --random_restarts 10
```

### Comparison with Scipy

```bash
# Compare on a single instance
python test_2opt_vs_scipy.py --instance tai10a.dat

# Compare on multiple small instances
python test_2opt_vs_scipy.py --max_instances 5 --max_size 15
```

### Demo

```bash
# Run the demonstration
python demo_2opt.py
```

## Performance Results

Based on testing with QAP instances from QAPLIB:

| Instance | Size | Scipy Objective | 2-opt Objective | Improvement | 2-opt Gap |
|----------|------|----------------|-----------------|-------------|-----------|
| tai10a   | 10×10| 157,954        | 135,640        | 14.13%      | N/A       |
| nug12    | 12×12| 596            | 588            | 1.34%       | 1.73%     |
| rou12    | 12×12| 245,168        | 241,122        | 1.65%       | 2.38%     |
| tai12a   | 12×12| 244,672        | 235,704        | 3.67%       | 5.03%     |
| had12    | 12×12| 1,674          | 1,656          | 1.08%       | 0.24%     |
| tai15a   | 15×15| 397,376        | 393,734        | 0.92%       | 1.42%     |

**Summary**: 
- 2-opt consistently outperforms scipy's solver in solution quality
- Average improvement: 1-14% better objective values
- Better optimality gaps when known optimal solutions exist
- Runtime: 40-80x slower than scipy but still very fast (< 1 second for small instances)

## Algorithm Parameters

- **`max_iterations`**: Maximum iterations per restart (default: 1000)
- **`random_restarts`**: Number of random starting points (default: 5)
- **`seed`**: Random seed for reproducibility (default: None)

## Key Features

1. **Efficient Delta Calculation**: Computes the change in objective without full recalculation
2. **Multiple Restarts**: Escapes local optima through random restarts
3. **Scalable**: Works well on instances up to 50×50 (larger instances may require parameter tuning)
4. **Verification**: Built-in solution verification to ensure correctness

## Implementation Details

### Delta Calculation

The most critical optimization is the efficient calculation of the objective change when swapping facilities. Instead of recalculating the entire objective (O(n²)), we compute only the affected terms (O(n)).

For swapping facilities at positions `i` and `j`:
```python
delta = sum over k≠i,j of:
    D[i,k] × (F[new_fac_i, fac_k] - F[old_fac_i, fac_k]) + 
    D[k,i] × (F[fac_k, new_fac_i] - F[fac_k, old_fac_i]) +
    D[j,k] × (F[new_fac_j, fac_k] - F[old_fac_j, fac_k]) +
    D[k,j] × (F[fac_k, new_fac_j] - F[fac_k, old_fac_j])
```

### Time Complexity

- **Per iteration**: O(n³) - trying all O(n²) swaps, each requiring O(n) delta calculation
- **Total**: O(n³ × iterations × restarts)

## Requirements

```bash
pip install numpy scipy
```

## Testing Environment

To test the implementation, first enter the conda environment:

```bash
bash  # Enter bash shell
cd /path/to/BE-for-Combinatorial-Optimization-
python qap/qap_2opt.py --help
```

## References

1. Croes, G. A. (1958). A method for solving traveling-salesman problems. Operations Research, 6(6), 791-812.
2. Burkard, R. E., Karisch, S. E., & Rendl, F. (1997). QAPLIB–a quadratic assignment problem library. Journal of Global optimization, 10(4), 391-403.
3. Loiola, E. M., et al. (2007). A survey for the quadratic assignment problem. European Journal of Operational Research, 176(2), 657-690.

## Future Improvements

- **First-improvement vs Best-improvement**: Currently uses first-improvement strategy
- **Tabu Search**: Add memory to avoid cycling between solutions  
- **Variable Neighborhood Search**: Combine with other neighborhood structures
- **Parallel Processing**: Parallelize the restart strategy
- **Adaptive Parameters**: Adjust parameters based on problem characteristics