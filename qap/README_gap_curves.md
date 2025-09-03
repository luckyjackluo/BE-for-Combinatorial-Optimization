# QAP Gap Curve Analysis

This document describes the new gap curve analysis functionality for the QAP (Quadratic Assignment Problem) optimization.

## Overview

The gap curve analysis focuses on tracking the percentage gap from the optimal solution during optimization, rather than tracking soft loss and hard loss separately. This provides a more direct measure of solution quality.

## Changes Made

### 1. Modified `test_grad_be_qap.py`

- **Data Collection**: Now records gap percentage instead of soft/hard loss
- **JSON Structure**: Simplified to focus on gap metrics
- **File Naming**: Changed from `loss_curves_*.json` to `gap_curves_*.json`

### 2. Enhanced `read_qap_log.py`

- **New Functions**: Added gap curve specific plotting functions
- **Backward Compatibility**: Maintained support for old format files
- **Multi-problem Analysis**: Functions for analyzing multiple problems and runs

## Data Format

### New Gap Curve JSON Format

```json
{
  "problem_name.dat": {
    "1": [
      {
        "epoch": 0,
        "gap": 15.23,
        "optimal_value": 1000,
        "current_tl": 1152.3,
        "best_tl": 1152.3,
        "elapsed_time": 0.5
      },
      ...
    ],
    "2": [...],
    "3": [...]
  }
}
```

### Key Fields

- `epoch`: Optimization iteration number
- `gap`: Percentage gap from optimal value (e.g., 15.23 means 15.23% above optimal)
- `optimal_value`: Known optimal solution value (if available)
- `current_tl`: Current total loss value
- `best_tl`: Best total loss found so far
- `elapsed_time`: Time elapsed since start of run

## New Functions

### 1. `read_gap_curves_from_json(json_path)`

Reads gap curve data from JSON file.

```python
from read_qap_log import read_gap_curves_from_json

gap_results = read_gap_curves_from_json('train_be/gap_curves_gd+k+5_real_None_constant_20240503_221500.json')
```

### 2. `plot_gap_curves_for_run(results, problem_name, run_index, title=None)`

Plots gap curves for a specific problem and run.

```python
plot_gap_curves_for_run(gap_results, 'qap_n20_d1.00_id0.dat', 1)
```

### 3. `plot_best_run_gap_curves_for_problem(results, problem_name, title=None)`

Plots gap curves for the best run (lowest final gap) of a problem.

```python
plot_best_run_gap_curves_for_problem(gap_results, 'qap_n20_d1.00_id0.dat')
```

### 4. `plot_average_gap_curves_across_problems(results, problem_names=None, title=None)`

Plots average gap curves across multiple problems using the best run from each.

```python
plot_average_gap_curves_across_problems(gap_results, ['qap_n20_d1.00_id0.dat', 'qap_n20_d1.00_id1.dat'])
```

### 5. `plot_problem_gap_comparison(results, problem_names, run_indices=None, title=None)`

Compares gap curves from different problems on the same plot.

```python
plot_problem_gap_comparison(gap_results, ['qap_n20_d1.00_id0.dat', 'qap_n20_d1.00_id1.dat'])
```

## Usage Examples

### Basic Usage

```python
# Read gap curves
gap_results = read_gap_curves_from_json('train_be/gap_curves_gd+k+5_real_None_constant_20240503_221500.json')

# Plot single run
plot_gap_curves_for_run(gap_results, 'qap_n20_d1.00_id0.dat', 1)

# Plot best run for a problem
plot_best_run_gap_curves_for_problem(gap_results, 'qap_n20_d1.00_id0.dat')

# Plot average across problems
problem_names = ['qap_n20_d1.00_id0.dat', 'qap_n20_d1.00_id1.dat']
plot_average_gap_curves_across_problems(gap_results, problem_names)
```

### Advanced Analysis

Use the provided example scripts:

1. **`test_gap_curves.py`**: Basic demonstration of gap curve functions
2. **`example_gap_analysis.py`**: Comprehensive analysis with multiple scenarios

```bash
# Run basic test
python test_gap_curves.py

# Run comprehensive analysis
python example_gap_analysis.py
```

## Running the Optimization

To generate gap curve data, run the optimization script:

```bash
python test_grad_be_qap.py --dataset real --alg gd+k+5 --lr 0.01 --device_idx 0 --S constant
```

This will create a file like:
`train_be/gap_curves_gd+k+5_real_None_constant_20240503_221500.json`

## Key Features

1. **Gap-focused Analysis**: Direct measurement of solution quality relative to optimal
2. **Multi-run Support**: Automatically selects best run for each problem
3. **Multi-problem Averaging**: Aggregates results across multiple problems
4. **Visualization**: Clear plots showing gap reduction over time
5. **Backward Compatibility**: Can still read old format files

## Interpretation

- **Lower Gap = Better**: Gap percentage shows how far the solution is from optimal
- **Convergence**: Gap should decrease over epochs as optimization progresses
- **Best Run Selection**: For each problem, the run with lowest final gap is selected
- **Average Performance**: Multi-problem analysis shows overall algorithm performance

## File Structure

```
qap/
├── test_grad_be_qap.py          # Main optimization script (modified)
├── read_qap_log.py              # Analysis functions (enhanced)
├── test_gap_curves.py           # Basic usage example
├── example_gap_analysis.py      # Comprehensive analysis example
├── README_gap_curves.md         # This documentation
└── train_be/                    # Output directory
    └── gap_curves_*.json        # Generated gap curve data
``` 