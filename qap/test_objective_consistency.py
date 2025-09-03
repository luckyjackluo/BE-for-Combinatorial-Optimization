#!/usr/bin/env python3
"""
Test script to verify that both solve_qap_scipy.py and solve_qap_grad_be.py 
use the same objective function calculation.
"""

import numpy as np
import torch
import sys
import os

# Import functions from both modules
sys.path.append('.')
from qap.solve_qap_scipy import compute_qap_cost_torch as scipy_torch, compute_qap_cost_numpy as scipy_numpy
from qap.solve_qap_grad_be import compute_qap_cost_torch as gradbe_torch, compute_qap_cost_numpy as gradbe_numpy, verify_objective_consistency


def test_objective_consistency():
    """Test that all objective function implementations give the same result."""
    
    print("Testing QAP objective function consistency...")
    print("="*60)
    
    # Create a small test problem
    n = 5
    np.random.seed(42)
    
    # Generate random distance and flow matrices
    dist_matrix = np.random.randint(1, 20, (n, n))
    flow_matrix = np.random.randint(1, 15, (n, n))
    
    # Make symmetric for realism (though not required)
    dist_matrix = (dist_matrix + dist_matrix.T) // 2
    flow_matrix = (flow_matrix + flow_matrix.T) // 2
    
    # Generate a random permutation
    permutation = np.random.permutation(n)
    
    # Create permutation matrix
    P = np.zeros((n, n))
    for i, j in enumerate(permutation):
        P[i, j] = 1.0
    
    print(f"Test problem size: {n}x{n}")
    print(f"Distance matrix:\n{dist_matrix}")
    print(f"Flow matrix:\n{flow_matrix}")
    print(f"Permutation: {permutation}")
    print(f"Permutation matrix:\n{P}")
    print()
    
    # Convert to torch tensors
    A_torch = torch.tensor(dist_matrix, dtype=torch.float32)
    B_torch = torch.tensor(flow_matrix, dtype=torch.float32)
    P_torch = torch.tensor(P, dtype=torch.float32)
    
    # Test all implementations
    results = {}
    
    # Scipy torch implementation
    try:
        results['scipy_torch'] = scipy_torch(A_torch, B_torch, P_torch).item()
        print(f"✓ scipy_torch result: {results['scipy_torch']:.6f}")
    except Exception as e:
        print(f"✗ scipy_torch failed: {e}")
        results['scipy_torch'] = None
    
    # Scipy numpy implementation  
    try:
        results['scipy_numpy'] = scipy_numpy(dist_matrix, flow_matrix, permutation)
        print(f"✓ scipy_numpy result: {results['scipy_numpy']:.6f}")
    except Exception as e:
        print(f"✗ scipy_numpy failed: {e}")
        results['scipy_numpy'] = None
    
    # Grad_be torch implementation
    try:
        results['gradbe_torch'] = gradbe_torch(A_torch, B_torch, P_torch).item()
        print(f"✓ gradbe_torch result: {results['gradbe_torch']:.6f}")
    except Exception as e:
        print(f"✗ gradbe_torch failed: {e}")
        results['gradbe_torch'] = None
    
    # Grad_be numpy implementation
    try:
        results['gradbe_numpy'] = gradbe_numpy(dist_matrix, flow_matrix, permutation)
        print(f"✓ gradbe_numpy result: {results['gradbe_numpy']:.6f}")
    except Exception as e:
        print(f"✗ gradbe_numpy failed: {e}")
        results['gradbe_numpy'] = None
    
    # Grad_be verification function
    try:
        verification = verify_objective_consistency(A_torch, B_torch, P_torch, permutation)
        print(f"✓ verification function:")
        print(f"  torch: {verification['torch_result']:.6f}")
        print(f"  numpy: {verification['numpy_result']:.6f}")
        print(f"  diff: {verification['difference']:.8f}")
        print(f"  consistent: {verification['consistent']}")
    except Exception as e:
        print(f"✗ verification function failed: {e}")
    
    print()
    print("Consistency Analysis:")
    print("-" * 40)
    
    # Filter out None results
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if len(valid_results) < 2:
        print("Not enough valid results to compare!")
        return False
    
    # Compare all pairs
    result_values = list(valid_results.values())
    result_names = list(valid_results.keys())
    
    all_consistent = True
    tolerance = 1e-6
    
    for i in range(len(result_values)):
        for j in range(i + 1, len(result_values)):
            diff = abs(result_values[i] - result_values[j])
            consistent = diff < tolerance
            
            print(f"{result_names[i]} vs {result_names[j]}: diff = {diff:.8f} ({'✓' if consistent else '✗'})")
            
            if not consistent:
                all_consistent = False
    
    print()
    if all_consistent:
        print("🎉 All objective function implementations are CONSISTENT!")
        print(f"Common result: {result_values[0]:.6f}")
    else:
        print("❌ Objective function implementations are INCONSISTENT!")
        print("This needs to be fixed before comparing solver results.")
    
    return all_consistent


def manual_calculation_check():
    """Do a manual calculation to verify the QAP objective formula."""
    print("\nManual Calculation Verification:")
    print("="*40)
    
    # Very simple 2x2 example for manual verification
    n = 2
    A = np.array([[0, 1], [1, 0]], dtype=float)  # Distance matrix
    B = np.array([[0, 2], [2, 0]], dtype=float)  # Flow matrix  
    perm = np.array([1, 0])  # Swap facilities: facility 0->location 1, facility 1->location 0
    
    print(f"Simple 2x2 example:")
    print(f"Distance matrix A:\n{A}")
    print(f"Flow matrix B:\n{B}")
    print(f"Permutation: {perm}")
    
    # Manual calculation
    # QAP objective = sum_i sum_j A[i,j] * B[perm[i], perm[j]]
    manual_result = 0
    for i in range(n):
        for j in range(n):
            term = A[i, j] * B[perm[i], perm[j]]
            manual_result += term
            print(f"A[{i},{j}] * B[{perm[i]},{perm[j]}] = {A[i,j]} * {B[perm[i], perm[j]]} = {term}")
    
    print(f"Manual calculation result: {manual_result}")
    
    # Test with our functions
    P_matrix = np.array([[0, 1], [1, 0]], dtype=float)
    A_torch = torch.tensor(A, dtype=torch.float32)
    B_torch = torch.tensor(B, dtype=torch.float32)
    P_torch = torch.tensor(P_matrix, dtype=torch.float32)
    
    numpy_result = scipy_numpy(A, B, perm)
    torch_result = scipy_torch(A_torch, B_torch, P_torch).item()
    
    print(f"Numpy function result: {numpy_result}")
    print(f"Torch function result: {torch_result}")
    
    print(f"Manual vs Numpy: diff = {abs(manual_result - numpy_result):.8f}")
    print(f"Manual vs Torch: diff = {abs(manual_result - torch_result):.8f}")


if __name__ == "__main__":
    try:
        consistent = test_objective_consistency()
        manual_calculation_check()
        
        if consistent:
            print("\n✅ All tests passed! Both solvers use consistent objective functions.")
            sys.exit(0)
        else:
            print("\n❌ Tests failed! Objective functions are inconsistent.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)