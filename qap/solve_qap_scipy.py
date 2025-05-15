import numpy as np
import os
import time
from scipy.optimize import quadratic_assignment
import logging
from datetime import datetime
import argparse

# Set up logging
def setup_logging(dataset_type):
    log_filename = f'qap_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{dataset_type}_faq.log'
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
            sol_file = file_path.replace('prob', 'sol').replace('.dat', '.sln')
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
        
        # Calculate objective value
        objective_value = result.fun
        
        # Calculate gap if optimal value is available
        gap = None
        if optimal_value is not None:
            gap = ((objective_value - optimal_value) / optimal_value) * 100
        
        return {
            'success': True,
            'objective_value': objective_value,
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
        prob_dir = 'qap/prob'
    else:  # synthetic
        prob_dir = 'qap/synthetic'
    
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
                logging.info(f"Objective value: {result['objective_value']}")
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
