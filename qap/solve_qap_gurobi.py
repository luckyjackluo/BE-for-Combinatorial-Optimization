import gurobipy as gp
from gurobipy import GRB
import numpy as np
import os
import time
import logging
import signal
from datetime import datetime

# Set Gurobi license file location
os.environ['GRB_LICENSE_FILE'] = '/data/zhishang/be/gurobi.lic'

# Set up logging
log_filename = f'qap_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}_gurobi.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Process timed out")

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
        # Split the line into numbers and convert to integers
        numbers = list(map(int, lines[current_line].split()))
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
        # Split the line into numbers and convert to integers
        numbers = list(map(int, lines[current_line].split()))
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

def solve_qap_instance(file_path, time_limit=None):
    """Solve a QAP instance using Gurobi and return results."""
    try:
        # Read the instance
        dist_matrix, flow_matrix = read_qap_instance(file_path)
        
        # Get optimal solution if available
        sol_file = file_path.replace('prob', 'sol').replace('.dat', '.sln')
        optimal_value = read_optimal_solution(sol_file)
        
        # Record start time
        start_time = time.time()
        
        # Get problem size
        n = len(dist_matrix)
        
        # Set time limit based on problem size, this is already much larger than BE time limit
        if time_limit is None:
            time_limit = n * 2 + 20
            
        # Set hard timeout limit (n*10 seconds)
        hard_timeout = n * 6
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(hard_timeout)
        
        try:
            # Create Gurobi model
            model = gp.Model("QAP")
            
            # Create binary variables for assignments
            x = model.addVars(n, n, vtype=GRB.BINARY, name="assign")
            
            # Assignment constraints
            model.addConstrs((x.sum(i, '*') == 1 for i in range(n)), "facility")
            model.addConstrs((x.sum('*', j) == 1 for j in range(n)), "location")
            
            # ------------------------------------------------------------------
            # Compact MIQP objective:  x^T  (F ⊗ D)  x
            #   where x is the n^2-vector of assignment binaries and ⊗ is the
            #   Kronecker product.  This avoids constructing n^4 Python objects
            #   and scales to n≈100 easily.
            # ------------------------------------------------------------------

            # 1. Flatten the (i,j) variables into a 1-D NumPy array for easy
            #    indexing that matches the Kronecker layout.
            x_vec = np.array([x[i, j] for i in range(n) for j in range(n)],
                              dtype=object)  # dtype=object to store GRBVar

            # 2. Build the coefficient matrix Q = F ⊗ D (NumPy is C-backed → fast)
            Q = np.kron(flow_matrix, dist_matrix)  # shape (n^2, n^2)

            # 3. Grab the upper-triangle indices once
            n2 = n * n
            triu_r, triu_c = np.triu_indices(n2)
            coeffs = Q[triu_r, triu_c]

            # 4. Filter out zero coefficients to keep the expression compact
            nz_mask = coeffs != 0
            coeffs = coeffs[nz_mask]
            triu_r = triu_r[nz_mask]
            triu_c = triu_c[nz_mask]

            # 5. Double the off-diagonal coefficients because we only keep
            #    the upper triangle (except diagonal).  This reproduces the
            #    full x^T Q x value.
            off_diag = triu_r != triu_c
            coeffs = coeffs * (2 * off_diag + (~off_diag))  # multiply by 2 if off-diag

            # 6. Build the quadratic expression with one addTerms call
            quad_obj = gp.QuadExpr()
            quad_obj.addTerms(coeffs.tolist(),
                              x_vec[triu_r].tolist(),
                              x_vec[triu_c].tolist())

            model.setObjective(quad_obj, GRB.MINIMIZE)
            
            # Set time limit
            model.setParam('TimeLimit', time_limit)
            
            # Log the time limit being used
            logging.info(f"Time limit set to {time_limit} seconds (n={n})")
            logging.info(f"Hard timeout set to {hard_timeout} seconds")
            
            # Optimize
            model.optimize()
            
            # Calculate runtime
            runtime = time.time() - start_time
            
            # Disable the alarm
            signal.alarm(0)
            
            if model.status == GRB.OPTIMAL:
                # Get solution
                solution = np.zeros(n, dtype=int)
                for i in range(n):
                    for j in range(n):
                        if x[i,j].X > 0.5:  # If x[i,j] is 1
                            solution[i] = j
                
                return {
                    'success': True,
                    'objective_value': model.objVal,
                    'optimal_value': optimal_value,
                    'gap': ((model.objVal - optimal_value) / optimal_value * 100) if optimal_value is not None else None,
                    'runtime': runtime,
                    'solution': solution,
                    'status': 'Optimal'
                }
            else:
                return {
                    'success': False,
                    'error': f"Gurobi status: {model.status}",
                    'runtime': runtime,
                    'status': model.status
                }
                
        except TimeoutError:
            # Disable the alarm
            signal.alarm(0)
            return {
                'success': False,
                'error': f"Process timed out after {hard_timeout} seconds",
                'runtime': time.time() - start_time,
                'status': 'Timeout'
            }
            
    except Exception as e:
        # Make sure to disable the alarm in case of any other exception
        signal.alarm(0)
        return {
            'success': False,
            'error': str(e)
        }

def main():
    # Get all .dat files from the qap/prob directory
    prob_dir = 'qap/input_data/prob'
    instance_files = [f for f in os.listdir(prob_dir) if f.endswith('.dat')]
    
    # Sort files by size (smaller instances first)
    instance_files.sort(key=lambda x: os.path.getsize(os.path.join(prob_dir, x)))
    
    for instance_file in instance_files:
        file_path = os.path.join(prob_dir, instance_file)
        logging.info(f"\nProcessing instance: {instance_file}")
        
        try:
            # Read and solve the instance
            result = solve_qap_instance(file_path)
            
            if result['success']:
                logging.info(f"Status: {result['status']}")
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
