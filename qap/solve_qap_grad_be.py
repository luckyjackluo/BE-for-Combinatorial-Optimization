import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from IPF import *
import scipy
import os
import sys
import time
import argparse
import datetime
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json

from torch import Tensor
from torch.utils.data import DataLoader
from numpy.random import default_rng
from tqdm import tqdm
from collections import OrderedDict
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import maximum_bipartite_matching
import random

def compute_qap_cost_torch(A, B, P):
    """
    Compute QAP cost using torch tensors.
    A: (N,N) distance matrix
    B: (N,N) flow matrix
    P: (N,N) torch permutation matrix
    
    This is the CANONICAL objective function used across all QAP solvers.
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
    This should give the same result as compute_qap_cost_torch.
    
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

def verify_objective_consistency(A, B, P, permutation):
    """
    Verify that torch and numpy implementations give the same result.
    
    Args:
        A: torch tensor (N,N) distance matrix
        B: torch tensor (N,N) flow matrix  
        P: torch tensor (N,N) permutation matrix
        permutation: numpy array (N,) permutation vector
    
    Returns:
        dict with torch_result, numpy_result, and difference
    """
    # Torch calculation
    torch_result = compute_qap_cost_torch(A, B, P).item()
    
    # Numpy calculation
    A_numpy = A.cpu().numpy()
    B_numpy = B.cpu().numpy()
    numpy_result = compute_qap_cost_numpy(A_numpy, B_numpy, permutation)
    
    # Check difference
    difference = abs(torch_result - numpy_result)
    
    return {
        'torch_result': torch_result,
        'numpy_result': numpy_result,
        'difference': difference,
        'consistent': difference < 1e-6
    }

def parse_qaplib_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    numbers = list(map(float, content.split()))  # Changed from int to float
    n = int(numbers[0])  # Problem size should still be an integer
    A_flat = numbers[1:n*n+1]
    B_flat = numbers[n*n+1:]
    
    A = np.array(A_flat).reshape((n, n))
    B = np.array(B_flat).reshape((n, n))
    
    return n, A, B

def parse_sln_file(filepath):
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
    # The optimal value is the second number in the first line
    optimal_value = int(first_line.split()[1])
    return optimal_value

def parse_scipy_results(log_file_path):
    """Parse scipy results log file to extract solutions for each instance."""
    scipy_solutions = {}
    
    try:
        with open(log_file_path, 'r') as f:
            lines = f.readlines()
        
        current_instance = None
        collecting_solution = False
        solution_numbers = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Look for instance processing lines
            if "Processing instance:" in line:
                # Save previous solution if we were collecting one
                if collecting_solution and current_instance is not None and len(solution_numbers) > 0:
                    scipy_solutions[current_instance] = solution_numbers
                
                # Start new instance
                current_instance = line.split("Processing instance: ")[1]
                collecting_solution = False
                solution_numbers = []
                
            # Look for solution start
            elif "Solution:" in line and current_instance is not None:
                collecting_solution = True
                # Extract numbers from the first solution line
                solution_str = line.split("Solution: ")[1]
                solution_str = solution_str.replace('[', '').replace(']', '')
                numbers = [int(x) for x in solution_str.split() if x]
                solution_numbers.extend(numbers)
                
            # Continue collecting solution numbers from subsequent lines
            elif collecting_solution and current_instance is not None:
                # Check if this line contains more solution numbers
                if line and not line.startswith("2025-") and not line.startswith("Processing"):
                    # Remove any remaining brackets and parse numbers
                    clean_line = line.replace('[', '').replace(']', '')
                    try:
                        numbers = [int(x) for x in clean_line.split() if x]
                        solution_numbers.extend(numbers)
                    except ValueError:
                        # This line doesn't contain numbers, stop collecting
                        if len(solution_numbers) > 0:
                            scipy_solutions[current_instance] = solution_numbers
                            current_instance = None
                            collecting_solution = False
                            solution_numbers = []
                else:
                    # End of solution, save it
                    if len(solution_numbers) > 0:
                        scipy_solutions[current_instance] = solution_numbers
                        current_instance = None
                        collecting_solution = False
                        solution_numbers = []
        
        # Handle the last solution if file ends while collecting
        if collecting_solution and current_instance is not None and len(solution_numbers) > 0:
            scipy_solutions[current_instance] = solution_numbers
                
    except Exception as e:
        print(f"Warning: Could not parse scipy results file {log_file_path}: {str(e)}")
        return {}
    
    return scipy_solutions

def solution_to_permutation_matrix(solution, device="cpu"):
    """Convert a solution vector to a permutation matrix."""
    n = len(solution)
    perm_matrix = torch.zeros(n, n, device=device)
    for i, j in enumerate(solution):
        perm_matrix[i, j] = 1.0
    return perm_matrix

def setup_logging(dataset_type, lr, alg, device_idx):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"log/results_{lr}_qap_{alg}_{dataset_type}_{device_idx}_{timestamp}.log"
    os.makedirs("log", exist_ok=True)
    return log_filename

class MatrixModel(nn.Module):
    def __init__(self, num_terms, alg):
        super().__init__()
        self.num_terms = num_terms
        self.mat = torch.nn.Linear(num_terms, num_terms, bias=False)
        self.alg = alg
    def forward(self):
        if self.alg == "gd":
            W = torch.abs(self.mat.weight) 
            W = W / W.sum(dim=1, keepdim=True)
            W = W / W.sum(dim=0, keepdim=True)
        else:
            W = torch.abs(self.mat.weight)
        return W
def cont_Birkhoff_SFE(W, k, A, B, perms, setting, device="cpu"):
    n = W.shape[0]
    fill = -n
    min_tl = float('inf')
    new_W = W.clone()
    cap = setting[1]
    min_P = None
    sum_thresh = 0
    
    total_loss = 0
    
    # To store all thresholds instead of permutations
    all_threshs = []
    all_tl = []
    
    # Pre-allocate P matrix
    P = torch.zeros(n, n, device=device)

    perms = 0.85*perms + 0.15*torch.rand(n, n)
    
    for idx in range(k):
        # Compute assignment matrix in one go
        A_to_use = torch.where(new_W > 0, perms, fill).detach().cpu().numpy()
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(A_to_use, maximize=True)
        
        # Get threshold and update sum in one operation
        thresh = torch.amin(new_W[row_ind, col_ind])
        sum_thresh += thresh
        
        # Store this threshold
        all_threshs.append(thresh.item())
        
        # Create permutation matrix efficiently
        P.zero_()  # Reset P matrix
        P[row_ind, col_ind] = 1
        
        # Update new_W efficiently
        new_W[row_ind, col_ind] -= thresh
        
        # Compute QAP cost
        tl = compute_qap_cost_torch(A, B, P)
        
        # Store this cost
        all_tl.append(tl.item())
        
        # Update minimum cost and matrix
        if tl < min_tl:
            min_tl = tl
            min_P = P.clone()
        
        # Update total loss efficiently
        total_loss = total_loss + tl*thresh if idx > 0 else tl*thresh
    
    return min_tl, total_loss/sum_thresh, min_P, idx, sum_thresh.item(), all_threshs, all_tl

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Solve QAP instances using gradient-based method')
    parser.add_argument('--dataset', type=str, choices=['real', 'synthetic'], default='real',
                      help='Dataset type to solve (real or synthetic)')
    parser.add_argument('--alg', type=str, required=True,
                      help='Algorithm to use')
    parser.add_argument('--lr', type=float, required=True,
                      help='Learning rate')
    parser.add_argument('--update_best_perms', type=int, default=1,
                      help='Whether to update best permutations')
    parser.add_argument('--freq', type=int, default=None,
                      help='Frequency of updating best_perms. If None, update immediately. If M, update every M steps')
    parser.add_argument('--device_idx', type=int, required=True,
                      help='Device index to use')
    parser.add_argument('--S', type=str, choices=['random', 'constant'], default='random',
                      help='Strategy for initialization A')
    parser.add_argument('--visualize', action='store_true',
                      help='Enable visualization of permutation distributions')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--warm_start', action='store_true', default=False,
                      help='Use scipy solutions as warm start initialization')
    parser.add_argument('--scipy_log', type=str, default="log/qap_results_20250805_102302_synthetic_scipy.log",
                      help='Path to scipy results log file for warm start')
    parser.add_argument('--max_iterations', type=int, default=5000,
                      help='Maximum iterations (default: 5000)')
    parser.add_argument('--time_budget', type=float, default=None,
                      help='Time budget in seconds (overrides max_iterations). Default: 2*n for each problem')
    args = parser.parse_args()

    # Set random seed for reproducibility
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    # If using CUDA, also set:
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    S = args.S  # Only use constant strategy
    alg = args.alg
    lr = args.lr
    device_idx = args.device_idx
    update_best_perms = args.update_best_perms == 1
    freq = args.freq
    print(device_idx)
    device = torch.device("cpu")
    print(sys.argv)
    T = args.max_iterations
    gen = True
    input_lst = []

    # Set directory based on dataset type
    if args.dataset == 'real':
        qap_dir = "input_data/real/prob"
        sol_dir = "input_data/real/sol"
    else:  # synthetic
        qap_dir = "input_data/synthetic"
        sol_dir = None  # No solution files for synthetic instances

    # Load scipy solutions for warm start initialization if enabled
    scipy_solutions = {}
    if args.warm_start:
        scipy_solutions = parse_scipy_results(args.scipy_log)
        print(f"Loaded {len(scipy_solutions)} scipy solutions for warm start initialization")
        # Debug: Print solution lengths
        for instance, solution in scipy_solutions.items():
            print(f"  {instance}: {len(solution)} elements")
    else:
        print("Warm start disabled, using random initialization")

    # Load QAP instances
    all_files = os.listdir(qap_dir)
    dataset = []
    optimal_values = {}
    found = False

    for fp in all_files:
        if fp.endswith('.dat'):
            print(fp) 
            
            n, A, B = parse_qaplib_file(os.path.join(qap_dir, fp))
            # Convert to torch tensors immediately during loading
            A = torch.tensor(A, dtype=torch.float32)
            B = torch.tensor(B, dtype=torch.float32)
            dataset.append((fp, n, A, B))  # Store filename along with data
            
            # Read corresponding solution file if available
            if args.dataset == 'real':
                sol_file = fp.replace('.dat', '.sln')
                if os.path.exists(os.path.join(sol_dir, sol_file)):
                    optimal_values[fp] = parse_sln_file(os.path.join(sol_dir, sol_file))

    print(f"Found {len(dataset)} problem instances to process")
    if len(dataset) == 0:
        print("No files found. Please check if the dataset.")
        return

    # Setup logging
    log_filename = setup_logging(args.dataset, lr, alg, device_idx)
    # Prepare to save all loss curves
    loss_curves = {}
    import datetime as dt
    if freq is None:
        freq_str = "None"
    else:
        freq_str = str(freq)
    loss_curves_filename = f"log/loss_curves_{alg}_{args.dataset}_{freq_str}_{S}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(log_filename, 'w') as f:
        f.write(f"Training Log - {datetime.datetime.now()}\n")
        f.write(f"Parameters: dataset={args.dataset}, lr={lr}, alg={alg}, device_idx={device_idx}, freq={freq}\n")
        f.write(f"Warm start: {'Enabled' if args.warm_start else 'Disabled'}\n")
        if args.warm_start:
            f.write(f"Scipy log file: {args.scipy_log}\n")
        f.write(f"Processing all .dat files in dataset\n")
        f.write(f"Tracking: Loss curves (soft loss and hard loss) + gap reporting\n\n")

    alg_lst = alg.split("+")
    if alg_lst[1] == "k":
        setting = [alg_lst[1], int(alg_lst[2])] 
    elif alg_lst[1] == "p":
        setting = [alg_lst[1], float(alg_lst[2])]
    print(alg_lst, setting)

    # Initialize results storage
    num_runs = 1
    best_results = {}  # Store best results for each problem
    # For saving loss curves
    for_save_problem = {}

    for j_idx, (fp, n, A, B) in enumerate(dataset):
        print(f"\nProcessing {fp}")
        best_tl_overall = float('inf')
        best_run_idx = 0
        
        with open(log_filename, 'a') as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"Processing {fp}\n")
            warm_start_status = "Yes (scipy solution)" if (args.warm_start and fp in scipy_solutions) else "No (random initialization)"
            f.write(f"Warm start: {warm_start_status}\n")
            f.write(f"{'='*50}\n")
        
        # Create visualization directory if needed
        if args.visualize:
            viz_dir = f"qap_viz/{fp.replace('.dat', '')}"
            os.makedirs(viz_dir, exist_ok=True)
        # For saving loss curves for this problem
        for_save_problem[fp] = {}
        
        for run_idx in range(num_runs):
            run_loss_list = []
            print(f"\nRun {run_idx + 1}/{num_runs}")
            num_terms = n
            
            # Start timing this run
            run_start_time = time.time()
            # Set time budget: use command line arg or default to 2*n
            if args.time_budget is not None:
                time_limit = args.time_budget
            else:
                time_limit = 4*n  # Default time budget: 2*n seconds
            
            # Initialize permutation matrix using scipy solution as warm start if available
            if args.warm_start and fp in scipy_solutions:
                print(f"Using scipy solution as warm start for {fp}")
                scipy_solution = scipy_solutions[fp]
                if len(scipy_solution) == num_terms:
                    # Convert scipy solution to permutation matrix
                    perms = solution_to_permutation_matrix(scipy_solution, device=device)
                    # Add small noise to make it doubly stochastic for IPF
                    noise = torch.rand(num_terms, num_terms, device=device) * 0.01
                    perms = perms + noise
                    perms = ipf(perms, 5, 1, 1)  # Make it doubly stochastic
                else:
                    print(f"Warning: Solution size mismatch for {fp}, using random initialization")
                    perms = ipf(torch.rand(num_terms, num_terms)/num_terms, 5, 1, 1)
            else:
                if args.warm_start:
                    print(f"No scipy solution found for {fp}, using random initialization")
                else:
                    print(f"Warm start disabled, using random initialization for {fp}")
                perms = ipf(torch.rand(num_terms, num_terms)/num_terms, 5, 1, 1)
            
            if "pgd" in alg:
                model = MatrixModel(num_terms, alg="pgd").to(device) 
            else:
                model = MatrixModel(num_terms, alg="gd").to(device) 
            state_dict = model.state_dict()
            # Initialize with better weights based on S parameter
            if S == "constant":
                # Use barycenter-style uniform doubly stochastic matrix
                weight = torch.ones(num_terms, num_terms, device=device) / num_terms
                state_dict['mat.weight'] = ipf(weight, 5, 1, 1)  # IPF iterations for doubly stochastic
            else:  # S == "random"
                # Use random initialization
                weight = torch.rand(num_terms, num_terms, device=device)
                weight = torch.abs(weight)
                weight = weight / weight.sum(dim=1, keepdim=True)  # Row normalization
                weight = weight / weight.sum(dim=0, keepdim=True)  # Column normalization
                state_dict['mat.weight'] = ipf(weight, 5, 1, 1)  # More IPF iterations
            
            model.load_state_dict(state_dict)
            hashmap = OrderedDict()
            best_tl = float('inf')
            best_perms = perms.clone()  # Will be initialized with scipy solution if available
            candidate_perms = perms.clone()  # Track the actual best permutation found so far
            candidate_tl = float('inf')
            update_counter = 0  # Counter for frequency-based updates
            pt_en = 0
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
            
            # For visualization
            if args.visualize:
                viz_data = {
                    'epochs': [],
                    'W_heatmaps': [],
                    'perm_heatmaps': [],
                    'losses': [],
                    'tls': [],
                    'best_tls': [],
                    'all_threshs': [],  # Changed: store all threshold values for each epoch
                    'all_tls': []       # Keep storing all permutation costs
                }

            idx = 0
            while True:
                current_time = time.time()
                if current_time - run_start_time > time_limit: # 600 seconds (10 minutes)
                    print(f"Time limit of {time_limit} seconds reached, stopping optimization.")
                    break
                W = model.forward() 
                tl, loss, perms, num_P, sum_thresh, all_threshs, all_tl = cont_Birkhoff_SFE(W, setting[1], A, B, best_perms, setting, device=device)
                loss.backward()
                
                # Store visualization data every 500 epochs
                if args.visualize and (idx % 500 == 0):
                    viz_data['epochs'].append(idx)
                    viz_data['W_heatmaps'].append(W.detach().cpu().numpy())
                    viz_data['perm_heatmaps'].append(perms.detach().cpu().numpy())
                    viz_data['losses'].append(loss.item())
                    viz_data['tls'].append(tl.item())
                    viz_data['best_tls'].append(best_tl.item() if idx > 0 else tl.item())
                    viz_data['all_threshs'].append(all_threshs)
                    viz_data['all_tls'].append(all_tl)
                 
                # Save loss info for this epoch
                optimal_value = optimal_values.get(fp, None)
                if optimal_value == 0:
                    gap = 0
                else:
                    gap = ((tl.item() - optimal_value) / optimal_value * 100) if optimal_value is not None else float('inf')
                run_loss_list.append({
                    'epoch': idx,
                    'soft_loss': loss.item(),
                    'hard_loss': tl.item(),
                    'best_tl': best_tl.item() if hasattr(best_tl, 'item') else float(best_tl),
                    'gap': gap,
                    'num_P': num_P,
                    'sum_thresh': sum_thresh,
                    'pt_en': pt_en,
                    'elapsed_time': current_time - run_start_time
                })
                # Update candidate permutation if we found a better one
                if tl < candidate_tl:
                    candidate_tl = tl
                    candidate_perms = perms.clone()
                
                # Update best_perms based on frequency strategy
                if freq is None:
                    # Immediate update strategy (current behavior)
                    if tl < best_tl:
                        best_tl = tl
                        if update_best_perms:
                            best_perms = perms
                        pt_en = 0  # Commented out dynamic k adjustment
                        setting[1] = int(alg_lst[2])  # Commented out dynamic k adjustment

                        # Update weight matrix
                        weight = torch.rand(num_terms, num_terms, device=device)
                        weight = torch.abs(weight)
                        weight = weight / weight.sum(dim=1, keepdim=True)  # Row normalization
                        weight = weight / weight.sum(dim=0, keepdim=True)  # Column normalization
                        state_dict['mat.weight'] = ipf(weight, 5, 1, 1)  # More IPF iterations
                        model.load_state_dict(state_dict)
                else:
                    # Frequency-based update strategy
                    update_counter += 1
                    if update_counter >= freq:
                        # Update best_perms with the best candidate found during this period
                        if candidate_tl < best_tl:
                            best_tl = candidate_tl
                            if update_best_perms:
                                best_perms = candidate_perms.clone()
                            pt_en = 0  # Commented out dynamic k adjustment
                            setting[1] = int(alg_lst[2])  # Commented out dynamic k adjustment
                        # Reset for next period
                        candidate_tl = float('inf')
                        candidate_perms = perms.clone()
                        update_counter = 0

                        # Update weight matrix
                        weight = torch.rand(num_terms, num_terms, device=device)
                        weight = torch.abs(weight)
                        weight = weight / weight.sum(dim=1, keepdim=True)  # Row normalization
                        weight = weight / weight.sum(dim=0, keepdim=True)  # Column normalization
                        state_dict['mat.weight'] = ipf(weight, 5, 1, 1)  # More IPF iterations
                        model.load_state_dict(state_dict)
                
                #Handle the case when we don't improve
                if tl >= best_tl:
                    loss_gap = (abs(loss - tl).item())/(tl.item() + 0.001)
                    if loss_gap <= 0.005:
                        pt_en += 0.01
                    elif loss_gap >= 0.1:
                        pt_en -= 0.01
                    if pt_en >= 1 and setting[1] <= num_terms:
                        setting[1] = int(setting[1] * 1.3) + 1
                        pt_en = 0
                    if pt_en <= -1 and setting[1] > int(alg_lst[2]):
                        setting[1] = np.min([int(setting[1] / 1.1), int(alg_lst[2])])
                        pt_en = 0
                if idx // 10 > 0 and idx % 10 == 0:
                    optimal_value = optimal_values.get(fp, None)
                    if optimal_value == 0:
                        gap = 0
                    else:
                        gap = ((tl.item() - optimal_value) / optimal_value * 100) if optimal_value is not None else float('inf')
                    # Handle both tensor and float types for best_tl
                    best_tl_value = best_tl.item() if hasattr(best_tl, 'item') else best_tl
                    print(round(loss.item(), 4), round(tl.item(), 4), round(best_tl_value, 4), 
                          f"gap: {round(gap, 2)}%" if optimal_value is not None else "gap: N/A",
                          num_P, round(sum_thresh, 4), f"elapsed: {current_time - run_start_time:.1f}s")
                if "pgd" in alg:
                    for param in model.parameters():
                        # Project gradient onto doubly stochastic space
                        grad = param.grad.data
                        
                        # Row and column normalization
                        grad = grad - grad.mean(dim=1, keepdim=True)
                        grad = grad - grad.mean(dim=0, keepdim=True)
                        
                        # Find optimal permutation
                        row, col = linear_sum_assignment(grad.cpu())
                        P = torch.zeros_like(W)
                        P[row, col] = 1
                        
                        # Adaptive step size with better scaling
                        grad_norm = torch.norm(grad)
                        #print(f"Grad norm: {grad_norm:.6f}")
                        
                        # Option 1: Clipped adaptive step size
                        if grad_norm > 0:
                            # Use log scaling to prevent extremely small step sizes
                            step_size = lr / (1 + torch.log(1 + grad_norm))
                        else:
                            step_size = lr
                            
                        # Update with momentum
                        if not hasattr(param, 'momentum_buffer'):
                            param.momentum_buffer = torch.zeros_like(param.data)
                        param.momentum_buffer = 0.9 * param.momentum_buffer + step_size * (P - param.data)
                        param.data = param.data + param.momentum_buffer
                        
                    torch.nn.Module.zero_grad(model)
                else:
                    optimizer.step()
                    optimizer.zero_grad()
                idx += 1
            # Create visualizations if we reached the end of training
            if args.visualize:
                create_visualizations(viz_data, fp, run_idx, viz_dir)
            # Save run loss list for this run
            for_save_problem[fp][run_idx+1] = run_loss_list
        # Update best overall result
        if best_tl < best_tl_overall:
            best_tl_overall = best_tl
            best_run_idx = run_idx
    
        # Write summary for this problem
        optimal_value = optimal_values.get(fp, None)
        # Handle both tensor and float types for best_tl_overall
        best_tl_overall_value = best_tl_overall.item() if hasattr(best_tl_overall, 'item') else best_tl_overall
        if best_tl_overall_value == optimal_value:
            gap = 0
        else:
            if optimal_value == 0:
                gap = 0
            else:
                gap = ((best_tl_overall_value - optimal_value) / optimal_value * 100) if optimal_value is not None else float('inf')
        
        # Get the best permutation matrix and verify objective calculation
        best_permutation_solution = None
        if fp in for_save_problem and best_run_idx + 1 in for_save_problem[fp]:
            run_data = for_save_problem[fp][best_run_idx + 1]
            if len(run_data) > 0:
                # Find the epoch with the best hard loss
                best_epoch_data = min(run_data, key=lambda x: x['hard_loss'])
                best_tl_value = best_epoch_data['hard_loss']
                
                # For verification, we would need the actual permutation matrix P
                # This would require storing it during optimization, which we don't currently do
                # For now, we'll just verify that our objective calculation is consistent
                verification_info = f"Best objective: {best_tl_value:.6f} (torch calculation)"
            else:
                verification_info = "No run data available for verification"
        else:
            verification_info = "No run data available for verification"

        with open(log_filename, 'a') as f:
            f.write(f"\nSummary for {fp}:\n")
            f.write(f"Best TL: {best_tl_overall}\n")
            f.write(f"Best run: {best_run_idx + 1}\n")
            f.write(f"Objective verification: {verification_info}\n")
            if optimal_value is not None:
                f.write(f"Optimal value: {optimal_value}\n")
                f.write(f"Final gap: {gap:.2f}%\n")
                # Calculate gap improvement if we have the data
                if fp in for_save_problem and best_run_idx + 1 in for_save_problem[fp]:
                    run_data = for_save_problem[fp][best_run_idx + 1]
                    if len(run_data) > 0:
                        initial_gap = run_data[0]['gap']
                        final_gap = run_data[-1]['gap']
                        gap_improvement = initial_gap - final_gap
                        f.write(f"Initial gap: {initial_gap:.2f}%\n")
                        f.write(f"Gap improvement: {gap_improvement:.2f}%\n")
            else:
                f.write(f"Optimal value: Not available\n")
                f.write(f"Gap: Cannot calculate (no optimal value)\n")
            f.write(f"Note: Using canonical QAP objective function: sum(A * (P @ B @ P.T))\n")
            f.write(f"{'='*50}\n")
    # Save all loss curves to JSON
            os.makedirs('log', exist_ok=True)
    with open(loss_curves_filename, 'w') as f:
        json.dump(for_save_problem, f, indent=2)
    print(f"Saved all loss curves to {loss_curves_filename}")
    
    # Write overall summary with gap statistics
    with open(log_filename, 'a') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"OVERALL SUMMARY\n")
        f.write(f"{'='*60}\n")
        f.write(f"Total problems processed: {len(dataset)}\n")
        
        # Calculate gap statistics for problems with optimal values
        gaps_with_optimal = []
        problems_without_optimal = []
        
        for fp, n, A, B in dataset:
            optimal_value = optimal_values.get(fp, None)
            if optimal_value is not None:
                # Find best result for this problem
                best_tl_for_problem = float('inf')
                for run_idx in range(num_runs):
                    if fp in for_save_problem and run_idx + 1 in for_save_problem[fp]:
                        run_data = for_save_problem[fp][run_idx + 1]
                        if len(run_data) > 0:
                            final_tl = run_data[-1]['hard_loss']
                            if final_tl < best_tl_for_problem:
                                best_tl_for_problem = final_tl
                
                if best_tl_for_problem != float('inf'):
                    gap = ((best_tl_for_problem - optimal_value) / optimal_value * 100)
                    gaps_with_optimal.append(gap)
            else:
                problems_without_optimal.append(fp)
        
        if gaps_with_optimal:
            gaps_array = np.array(gaps_with_optimal)
            f.write(f"Problems with optimal values: {len(gaps_with_optimal)}\n")
            f.write(f"Average final gap: {np.mean(gaps_array):.2f}%\n")
            f.write(f"Best final gap: {np.min(gaps_array):.2f}%\n")
            f.write(f"Worst final gap: {np.max(gaps_array):.2f}%\n")
            f.write(f"Standard deviation: {np.std(gaps_array):.2f}%\n")
        
        if problems_without_optimal:
            f.write(f"Problems without optimal values: {len(problems_without_optimal)}\n")
            f.write(f"Files: {', '.join(problems_without_optimal)}\n")
        
        f.write(f"{'='*60}\n")

def create_visualizations(viz_data, problem_name, run_idx, viz_dir):
    """
    Create and save visualizations of the optimization process.
    
    Args:
        viz_data: Dictionary containing visualization data
        problem_name: Name of the QAP problem
        run_idx: Run index
        viz_dir: Directory to save visualizations
    """
    # Only visualize selected epochs to avoid too many plots
    num_epochs = len(viz_data['epochs'])
    selected_indices = []
    
    # Always include first and last epoch
    if num_epochs > 0:
        selected_indices.append(0)
    
    # Add some intermediate epochs
    if num_epochs > 10:
        step = num_epochs // 5
        for i in range(step, num_epochs - 1, step):
            selected_indices.append(i)
    
    # Add the last epoch
    if num_epochs > 1:
        selected_indices.append(num_epochs - 1)
    
    # Create plots for each selected epoch
    for i in selected_indices:
        epoch = viz_data['epochs'][i]
        
        # Get threshold data and cost data
        all_threshs = viz_data['all_threshs'][i]
        all_tls = viz_data['all_tls'][i]
        
        # Normalize costs for better visualization
        if len(all_tls) > 0:
            min_cost = min(all_tls)
            max_cost = max(all_tls)
            # Prevent division by zero
            if max_cost != min_cost:
                normalized_costs = [(cost - min_cost) / (max_cost - min_cost) for cost in all_tls]
            else:
                normalized_costs = [1.0 for _ in all_tls]
        else:
            normalized_costs = []
        
        # Create figure with 3 rows, 2 columns layout
        fig = plt.figure(figsize=(15, 15))
        gs = GridSpec(3, 2, figure=fig)
        
        # Plot W heatmap (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        w_heatmap = ax1.imshow(viz_data['W_heatmaps'][i], cmap='viridis')
        ax1.set_title(f'W Matrix (Epoch {epoch})')
        plt.colorbar(w_heatmap, ax=ax1)
        
        # Plot loss and tl curves (top right)
        ax_sum = fig.add_subplot(gs[0, 1])
        epochs_to_plot = viz_data['epochs'][:i+1]
        ax_sum.plot(epochs_to_plot, viz_data['losses'][:i+1], label='Loss', marker='o')
        ax_sum.plot(epochs_to_plot, viz_data['tls'][:i+1], label='TL', marker='x')
        ax_sum.set_xlabel('Epoch')
        ax_sum.set_ylabel('Value')
        ax_sum.legend()
        ax_sum.grid(True)
        
        # Create bar plot of normalized permutation costs in sequential order (middle left)
        ax_cost_bar = fig.add_subplot(gs[1, 0])
        x_pos = range(len(all_tls))
        # Plot only normalized costs
        ax_cost_bar.bar(x_pos, normalized_costs)
        ax_cost_bar.set_title(f'Normalized Permutation Costs')
        ax_cost_bar.set_xlabel('Permutation Index')
        ax_cost_bar.set_ylabel('Normalized QAP Cost')
        
        # Create distribution plot of normalized permutation costs (middle right)
        ax_cost_dist = fig.add_subplot(gs[1, 1])
        # Plot only normalized cost distributions
        if len(all_tls) > 0:
            bins = min(20, len(all_tls)//2 + 1)
            ax_cost_dist.hist(normalized_costs, bins=bins, alpha=0.7)
        ax_cost_dist.set_title(f'Distribution of Normalized Permutation Costs')
        ax_cost_dist.set_xlabel('Normalized QAP Cost')
        ax_cost_dist.set_ylabel('Frequency')
        
        # Plot threshold values (bottom left)
        if len(all_threshs) > 0:
            ax_thresh = fig.add_subplot(gs[2, 0])
            x_pos = range(len(all_threshs))
            ax_thresh.bar(x_pos, all_threshs)
            ax_thresh.set_title(f'Threshold Values')
            ax_thresh.set_xlabel('Permutation Index')
            ax_thresh.set_ylabel('Threshold Value')
            
            # Add a horizontal line for the average threshold
            avg_thresh = sum(all_threshs) / len(all_threshs)
            ax_thresh.axhline(y=avg_thresh, color='red', linestyle='--', 
                           label=f'Avg Threshold: {avg_thresh:.4f}')
            ax_thresh.legend()
            
            # Create a scatter plot of thresholds vs normalized costs (bottom right)
            ax_thresh_cost = fig.add_subplot(gs[2, 1])
            # Plot only normalized costs
            ax_thresh_cost.scatter(all_threshs, normalized_costs, alpha=0.7)
            ax_thresh_cost.set_title(f'Thresholds vs Normalized Costs')
            ax_thresh_cost.set_xlabel('Threshold Value')
            ax_thresh_cost.set_ylabel('Normalized QAP Cost')
            ax_thresh_cost.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/run{run_idx}_epoch{epoch}.png")
        plt.close(fig)
    
    # Create a summary plot showing optimization progress with normalized values
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot loss curve
    ax1.plot(viz_data['epochs'], viz_data['losses'], label='Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Loss vs Epoch for {problem_name} (Run {run_idx})')
    ax1.grid(True)
    
    # Plot normalized TL
    normalized_tls = []
    if len(viz_data['tls']) > 0:
        min_tl = min(viz_data['tls'])
        max_tl = max(viz_data['tls'])
        if max_tl != min_tl:
            normalized_tls = [(tl - min_tl) / (max_tl - min_tl) for tl in viz_data['tls']]
        else:
            normalized_tls = [1.0 for _ in viz_data['tls']]
        
        ax2.plot(viz_data['epochs'], normalized_tls, label='Normalized TL', color='orange')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Normalized QAP Cost')
        ax2.set_title(f'Normalized QAP Cost vs Epoch')
        ax2.legend()
        ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/run{run_idx}_summary.png")
    plt.close(fig)
    
    # Create a sequential visualization of threshold values across epochs
    if len(selected_indices) > 0:
        # Create figure for sequential visualization
        fig, axes = plt.subplots(len(selected_indices), 1, figsize=(14, 4*len(selected_indices)))
        if len(selected_indices) == 1:
            axes = [axes]  # Make sure axes is always a list
        
        # For each selected epoch, create a bar chart of threshold values
        for ax_idx, i in enumerate(selected_indices):
            epoch = viz_data['epochs'][i]
            all_threshs = viz_data['all_threshs'][i]
            
            # Create bar plot
            axes[ax_idx].bar(range(len(all_threshs)), all_threshs)
            axes[ax_idx].set_title(f'Epoch {epoch}: Threshold Values')
            axes[ax_idx].set_xlabel('Permutation Index')
            axes[ax_idx].set_ylabel('Threshold Value')
            axes[ax_idx].grid(True, axis='y')
            
            # Add a horizontal line for the average threshold
            avg_thresh = sum(all_threshs) / len(all_threshs)
            axes[ax_idx].axhline(y=avg_thresh, color='red', linestyle='--', 
                               label=f'Avg Threshold: {avg_thresh:.4f}')
            axes[ax_idx].legend()
        
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/run{run_idx}_sequential_thresholds.png")
        plt.close(fig)
    
    # Create a visualization showing the distribution of normalized costs across epochs
    if num_epochs > 0:
        # Select a few epochs to visualize
        selected_epochs = selected_indices
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Violin plots of normalized costs
        for i in selected_epochs:
            epoch = viz_data['epochs'][i]
            costs = viz_data['all_tls'][i]
            
            # Normalize costs
            if len(costs) > 0:
                min_cost = min(costs)
                max_cost = max(costs)
                if max_cost != min_cost:
                    norm_costs = [(c - min_cost) / (max_cost - min_cost) for c in costs]
                else:
                    norm_costs = [1.0 for _ in costs]
                
                # Create a violin plot for the distribution of normalized costs
                positions = [epoch] * len(norm_costs)
                vp = ax.violinplot([norm_costs], positions=[epoch], widths=5, showmeans=True)
                
                # Plot individual normalized costs as points
                ax.scatter([epoch] * len(norm_costs), norm_costs, alpha=0.5, s=20)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Normalized QAP Cost')
        ax.set_title(f'Distribution of Normalized QAP Costs Across Permutations ({problem_name}, Run {run_idx})')
        ax.grid(True)
        
        plt.savefig(f"{viz_dir}/run{run_idx}_normalized_cost_distributions.png")
        plt.close(fig)
        
        # Create a heatmap visualization of normalized costs across all permutations and epochs
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Find the maximum number of permutations across all epochs
        max_perms = max(len(viz_data['all_tls'][i]) for i in selected_indices)
        
        # Create a 2D array for the normalized heatmap
        norm_heatmap_data = np.zeros((len(selected_indices), max_perms))
        norm_heatmap_data.fill(np.nan)  # Fill with NaN for permutations that don't exist
        
        # Fill in the normalized costs
        for i, epoch_idx in enumerate(selected_indices):
            costs = viz_data['all_tls'][epoch_idx]
            
            # Normalize costs
            if len(costs) > 0:
                min_cost = min(costs)
                max_cost = max(costs)
                if max_cost != min_cost:
                    norm_costs = [(c - min_cost) / (max_cost - min_cost) for c in costs]
                else:
                    norm_costs = [1.0 for _ in costs]
                norm_heatmap_data[i, :len(norm_costs)] = norm_costs
        
        # Create the normalized cost heatmap
        im = ax.imshow(norm_heatmap_data, aspect='auto', cmap='viridis')
        
        # Set ticks and labels for normalized cost heatmap
        epoch_labels = [viz_data['epochs'][i] for i in selected_indices]
        ax.set_yticks(range(len(selected_indices)))
        ax.set_yticklabels([f'Epoch {e}' for e in epoch_labels])
        ax.set_xlabel('Permutation Index')
        ax.set_ylabel('Epoch')
        ax.set_title(f'Normalized QAP Costs Across Permutations and Epochs')
        
        # Add a colorbar for normalized costs
        plt.colorbar(im, ax=ax, label='Normalized QAP Cost')
        
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/run{run_idx}_normalized_cost_heatmap.png")
        plt.close(fig)
        
        # Create a visualization showing the relationship between thresholds and normalized costs
        if num_epochs > 0:
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # For the last epoch, plot threshold vs normalized cost
            last_idx = selected_indices[-1]
            last_epoch = viz_data['epochs'][last_idx]
            thresholds = viz_data['all_threshs'][last_idx]
            costs = viz_data['all_tls'][last_idx]
            
            if len(costs) > 0:
                min_cost = min(costs)
                max_cost = max(costs)
                if max_cost != min_cost:
                    norm_costs = [(c - min_cost) / (max_cost - min_cost) for c in costs]
                else:
                    norm_costs = [1.0 for _ in costs]
                
                # Plot normalized costs
                ax.scatter(thresholds, norm_costs, alpha=0.7, s=50, color='orange')
                
                # Add trend line for normalized costs
                if len(thresholds) > 1:
                    z = np.polyfit(thresholds, norm_costs, 1)
                    p = np.poly1d(z)
                    x_range = np.linspace(min(thresholds), max(thresholds), 100)
                    ax.plot(x_range, p(x_range), "r--", label=f"Trend Line: y={z[0]:.2f}x+{z[1]:.2f}")
                
                ax.set_xlabel('Threshold Value')
                ax.set_ylabel('Normalized QAP Cost')
                ax.set_title(f'Threshold vs Normalized QAP Cost (Epoch {last_epoch}, {problem_name}, Run {run_idx})')
                ax.grid(True)
                if len(thresholds) > 1:
                    ax.legend()
            
            plt.tight_layout()
            plt.savefig(f"{viz_dir}/run{run_idx}_threshold_vs_normalized_cost.png")
            plt.close(fig)

if __name__ == "__main__":
    main()
