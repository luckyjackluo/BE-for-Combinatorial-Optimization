import numpy as np
import os
import time
from scipy.optimize import quadratic_assignment
import logging
from datetime import datetime
import argparse
import torch

def compute_qap_cost_torch(A, B, P):
    """
    Compute QAP cost using torch tensors.
    A: (N,N) distance matrix
    B: (N,N) flow matrix  
    P: (N,N) permutation matrix
    """
    # Ensure all tensors are float32
    A = A.to(dtype=torch.float32)
    B = B.to(dtype=torch.float32)
    P = P.to(dtype=torch.float32)
    PBPT = P @ B @ P.T   # P B P^T
    return torch.sum(A * PBPT)

def compute_qap_cost_numpy(dist_matrix, flow_matrix, permutation):
    """
    Compute QAP cost using numpy arrays.
    dist_matrix: (N,N) distance matrix
    flow_matrix: (N,N) flow matrix
    permutation: (N,) permutation vector where permutation[i] = facility assigned to location i
    """
    n = len(permutation)
    total_cost = 0.0
    
    for i in range(n):
        for j in range(n):
            total_cost += dist_matrix[i, j] * flow_matrix[permutation[i], permutation[j]]
    
    return total_cost

def permutation_to_matrix(permutation):
    """Convert permutation vector to permutation matrix."""
    n = len(permutation)
    P = np.zeros((n, n))
    for i, j in enumerate(permutation):
        P[i, j] = 1.0
    return P

# Set up logging
def setup_logging(dataset_type):
    log_filename = f'log/qap_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{dataset_type}_scipy.log'
    os.makedirs('log', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return log_filename

def read_qap_instance(file_path):
    """Read a QAP instance from a .dat file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Skip comments and empty lines
    lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
    
    # Get problem size
    n = int(lines[0])
    
    # Read distance and flow matrices
    dist_matrix = np.zeros((n, n))
    flow_matrix = np.zeros((n, n))
    
    # Read distance matrix
    current_line = 1
    current_row = 0
    current_numbers = []
    
    while current_row < n:
        # Split the line into numbers and convert to float
        numbers = list(map(float, lines[current_line].split()))
        current_numbers.extend(numbers)
        
        # If we have enough numbers for a complete row
        if len(current_numbers) >= n:
            dist_matrix[current_row] = current_numbers[:n]
            current_numbers = current_numbers[n:]
            current_row += 1
        
        current_line += 1
    
    # Read flow matrix
    current_row = 0
    current_numbers = []
    
    while current_row < n:
        # Split the line into numbers and convert to float
        numbers = list(map(float, lines[current_line].split()))
        current_numbers.extend(numbers)
        
        # If we have enough numbers for a complete row
        if len(current_numbers) >= n:
            flow_matrix[current_row] = current_numbers[:n]
            current_numbers = current_numbers[n:]
            current_row += 1
        
        current_line += 1
    
    return dist_matrix, flow_matrix

def read_optimal_solution(file_path):
    """Read the optimal solution from a .sln file."""
    try:
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
        # The optimal value is the second number in the first line
        optimal_value = int(first_line.split()[1])
        return optimal_value
    except Exception as e:
        logging.warning(f"Could not read optimal solution from {file_path}: {str(e)}")
        return None

def solve_qap_instance(file_path, dataset_type='real'):
    """Solve a QAP instance and return results."""
    try:
        # Read the instance
        dist_matrix, flow_matrix = read_qap_instance(file_path)
        
        # Get optimal solution if available
        if dataset_type == 'real':
            sol_file = file_path.replace('input_data/real/prob', 'input_data/real/sol').replace('.dat', '.sln')
        else:  # synthetic
            sol_file = None  # No optimal solutions for synthetic instances
        
        optimal_value = read_optimal_solution(sol_file) if sol_file else None
        
        # Record start time
        start_time = time.time()
        
        # Create initial permutation (identity permutation)
        n = len(dist_matrix)
        initial_perm = np.arange(n)
        
        # Solve using scipy's quadratic_assignment
        result = quadratic_assignment(dist_matrix, flow_matrix, method='faq', options={'initial_perm': initial_perm})
        
        # Calculate runtime
        runtime = time.time() - start_time
        
        # Calculate objective value using our consistent objective function
        # Use both scipy's result and our own calculation for verification
        scipy_objective = result.fun
        
        # Recalculate using our consistent objective function
        our_objective_numpy = compute_qap_cost_numpy(dist_matrix, flow_matrix, result.col_ind)
        
        # Also calculate using torch version for consistency with grad_be
        A_torch = torch.tensor(dist_matrix, dtype=torch.float32)
        B_torch = torch.tensor(flow_matrix, dtype=torch.float32)
        P_torch = torch.tensor(permutation_to_matrix(result.col_ind), dtype=torch.float32)
        our_objective_torch = compute_qap_cost_torch(A_torch, B_torch, P_torch).item()
        
        # Use our consistent calculation as the final objective
        objective_value = our_objective_numpy
        
        # Calculate gap if optimal value is available
        gap = None
        if optimal_value is not None:
            gap = ((objective_value - optimal_value) / optimal_value) * 100
        
        return {
            'success': True,
            'objective_value': objective_value,
            'scipy_objective': scipy_objective,
            'torch_objective': our_objective_torch,
            'optimal_value': optimal_value,
            'gap': gap,
            'runtime': runtime,
            'solution': result.col_ind
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Solve QAP instances')
    parser.add_argument('--dataset', type=str, choices=['real', 'synthetic'], default='real',
                      help='Dataset type to solve (real or synthetic)')
    args = parser.parse_args()
    
    # Setup logging
    log_filename = setup_logging(args.dataset)
    
    # Set directory based on dataset type
    if args.dataset == 'real':
        prob_dir = 'input_data/real/prob'
    else:  # synthetic
        prob_dir = 'input_data/synthetic'
    
    # Get all .dat files from the directory
    instance_files = [f for f in os.listdir(prob_dir) if f.endswith('.dat')]
    
    # Sort files by size (smaller instances first)
    instance_files.sort(key=lambda x: os.path.getsize(os.path.join(prob_dir, x)))
    
    for instance_file in instance_files:
        file_path = os.path.join(prob_dir, instance_file)
        logging.info(f"\nProcessing instance: {instance_file}")
        
        try:
            # Read and solve the instance
            result = solve_qap_instance(file_path, args.dataset)
            
            if result['success']:
                logging.info(f"Objective value (our calculation): {result['objective_value']}")
                logging.info(f"Scipy objective: {result['scipy_objective']}")
                logging.info(f"Torch objective: {result['torch_objective']}")
                
                # Check consistency between calculations
                diff_scipy = abs(result['objective_value'] - result['scipy_objective'])
                diff_torch = abs(result['objective_value'] - result['torch_objective'])
                logging.info(f"Difference from scipy: {diff_scipy:.6f}")
                logging.info(f"Difference from torch: {diff_torch:.6f}")
                
                if result['optimal_value'] is not None:
                    logging.info(f"Optimal value: {result['optimal_value']}")
                    logging.info(f"Gap: {result['gap']:.2f}%")
                else:
                    logging.info("No optimal value available for gap calculation")
                logging.info(f"Runtime: {result['runtime']:.2f} seconds")
                logging.info(f"Solution: {result['solution']}")
            else:
                logging.error(f"Failed to solve {instance_file}: {result['error']}")
                
        except Exception as e:
            logging.error(f"Error processing {instance_file}: {str(e)}")

if __name__ == "__main__":
    main() 
