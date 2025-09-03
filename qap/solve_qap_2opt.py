import numpy as np
import time
import random
from typing import Tuple, List, Optional
import os
import argparse
import logging
from datetime import datetime


def read_qap_instance(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read a QAP instance from a .dat file.
    
    Args:
        file_path: Path to the .dat file
        
    Returns:
        Tuple of (distance_matrix, flow_matrix)
    """
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


def read_optimal_solution(file_path: str) -> Optional[int]:
    """Read the optimal solution from a .sln file."""
    if not os.path.exists(file_path):
        return None
    
    try:
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
        # The optimal value is the second number in the first line
        optimal_value = int(first_line.split()[1])
        return optimal_value
    except Exception as e:
        logging.warning(f"Could not read optimal solution from {file_path}: {str(e)}")
        return None


def calculate_objective(solution: List[int], dist_matrix: np.ndarray, flow_matrix: np.ndarray) -> float:
    """Calculate the QAP objective value for a given solution."""
    n = len(solution)
    objective = 0.0
    for i in range(n):
        for j in range(n):
            objective += dist_matrix[i][j] * flow_matrix[solution[i]][solution[j]]
    return objective


def two_opt_qap(dist_matrix: np.ndarray, 
                flow_matrix: np.ndarray,
                max_iterations: int = 10000,
                time_budget_seconds: Optional[float] = None,
                neighborhood_size: int = 100,
                random_restarts: int = 1,
                seed: int = 42,
                verbose: bool = True) -> Tuple[List[int], float, int, int]:
    """
    Solve QAP using 2-opt local search.
    
    Args:
        dist_matrix: Distance matrix
        flow_matrix: Flow matrix
        max_iterations: Maximum iterations (ignored if time_budget_seconds is set)
        time_budget_seconds: Time budget in seconds (overrides max_iterations)
        neighborhood_size: Number of random swaps to try per iteration
        random_restarts: Number of random restarts
        seed: Random seed
        verbose: Whether to print progress information
        
    Returns:
        Tuple of (best_solution, best_objective, total_iterations, restarts_used)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    n = len(dist_matrix)
    best_solution = None
    best_objective = float('inf')
    total_iterations = 0
    
    start_time = time.time()
    
    # Determine whether to use time budget or iteration budget
    use_time_budget = time_budget_seconds is not None
    if use_time_budget:
        budget_seconds = time_budget_seconds
        print(f"Parameters: time_budget={budget_seconds}s, restarts={random_restarts}, neighborhood_size={neighborhood_size}")
    else:
        print(f"Parameters: max_iterations={max_iterations}, restarts={random_restarts}, neighborhood_size={neighborhood_size}")
    
    restarts_used = 0
    for restart in range(random_restarts):
        restarts_used = restart + 1
        
        # Check time budget for restarts
        if use_time_budget and time.time() - start_time >= budget_seconds:
            break
            
        # Generate random initial solution
        current_solution = list(range(n))
        random.shuffle(current_solution)
        current_objective = calculate_objective(current_solution, dist_matrix, flow_matrix)
        
        if verbose and random_restarts > 1:
            print(f"  Restart {restart + 1}/{random_restarts}: Initial objective = {current_objective:.0f}")
            logging.info(f"  Restart {restart + 1}/{random_restarts}: Initial objective = {current_objective:.0f}")
        
        iteration = 0
        restart_start_time = time.time()
        last_improvement_iter = 0
        improvements_count = 0
        
        while True:
            if use_time_budget:
                if time.time() - start_time >= budget_seconds:
                    break
            else:
                if iteration >= max_iterations:
                    break
            
            improved = False
            
            # Try neighborhood_size random swaps
            if neighborhood_size >= n * (n - 1) // 2:
                # If neighborhood size is large enough, try all pairs
                for i in range(n):
                    for j in range(i + 1, n):
                        # Try swapping positions i and j
                        new_solution = current_solution.copy()
                        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
                        new_objective = calculate_objective(new_solution, dist_matrix, flow_matrix)
                        
                        if new_objective < current_objective:
                            current_solution = new_solution
                            current_objective = new_objective
                            improved = True
                            improvements_count += 1
                            last_improvement_iter = iteration
                            if verbose:
                                elapsed = time.time() - restart_start_time
                                print(f"    Iter {iteration}: New objective = {current_objective:.0f} (improvement #{improvements_count}, {elapsed:.1f}s)")
                                logging.info(f"    Iteration {iteration}: Improved to {current_objective:.0f} (improvement #{improvements_count}, elapsed: {elapsed:.1f}s)")
                            break
                    if improved:
                        break
            else:
                for _ in range(neighborhood_size):
                    # Random swap
                    i, j = random.sample(range(n), 2)
                    
                    # Try swapping positions i and j
                    new_solution = current_solution.copy()
                    new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
                    new_objective = calculate_objective(new_solution, dist_matrix, flow_matrix)
                    
                    if new_objective < current_objective:
                        current_solution = new_solution
                        current_objective = new_objective
                        improved = True
                        improvements_count += 1
                        last_improvement_iter = iteration
                        if verbose:
                            elapsed = time.time() - restart_start_time
                            print(f"    Iter {iteration}: New objective = {current_objective:.0f} (improvement #{improvements_count}, {elapsed:.1f}s)")
                            logging.info(f"    Iteration {iteration}: Improved to {current_objective:.0f} (improvement #{improvements_count}, elapsed: {elapsed:.1f}s)")
                        break
            
            if not improved:
                if verbose:
                    elapsed = time.time() - restart_start_time
                    print(f"    Local optimum reached at iteration {iteration} (final: {current_objective:.0f}, {improvements_count} improvements, {elapsed:.1f}s)")
                    logging.info(f"    Local optimum reached at iteration {iteration}: {current_objective:.0f} ({improvements_count} improvements, elapsed: {elapsed:.1f}s)")
                break  # Local optimum reached
                
            iteration += 1
            total_iterations += 1
            
            # Periodic progress update (every 1000 iterations or every 30 seconds)
            if verbose and iteration > 0 and (iteration % 1000 == 0 or time.time() - restart_start_time > 30):
                elapsed = time.time() - restart_start_time
                time_since_improvement = iteration - last_improvement_iter
                if use_time_budget:
                    remaining_time = budget_seconds - (time.time() - start_time)
                    print(f"    Iter {iteration}: Current = {current_objective:.0f} ({improvements_count} improvements, {time_since_improvement} iters since last, {elapsed:.1f}s elapsed, {remaining_time:.1f}s remaining)")
                else:
                    print(f"    Iter {iteration}: Current = {current_objective:.0f} ({improvements_count} improvements, {time_since_improvement} iters since last, {elapsed:.1f}s elapsed)")
                logging.info(f"    Progress - Iteration {iteration}: Current objective = {current_objective:.0f}, {improvements_count} improvements found, {time_since_improvement} iterations since last improvement")
        
        # Update best solution
        if current_objective < best_objective:
            old_best = best_objective
            best_objective = current_objective
            best_solution = current_solution.copy()
            if verbose:
                improvement = old_best - best_objective if old_best != float('inf') else 0
                print(f"  New global best: {best_objective:.0f} (improvement: {improvement:.0f})")
                logging.info(f"  New global best objective: {best_objective:.0f} (improvement from previous: {improvement:.0f})")
    
    if verbose:
        total_time = time.time() - start_time
        print(f"Optimization complete: Best = {best_objective:.0f}, Total time = {total_time:.1f}s, Total iterations = {total_iterations}")
        logging.info(f"Optimization summary: Best objective = {best_objective:.0f}, Total time = {total_time:.1f}s, Total iterations = {total_iterations}, Restarts used = {restarts_used}")
    
    return best_solution, best_objective, total_iterations, restarts_used


def solve_qap_instance_2opt(file_path: str, 
                           dataset_type: str = 'real',
                           max_iterations: int = 10000,
                           time_budget_seconds: Optional[float] = None,
                           neighborhood_size: int = 100,
                           random_restarts: int = 1,
                           seed: int = 42,
                           verbose: bool = True) -> dict:
    """Solve a QAP instance using 2-opt and return results."""
    try:
        # Read the instance
        dist_matrix, flow_matrix = read_qap_instance(file_path)
        n = len(dist_matrix)
        print(f"Problem size: {n}x{n}")
        
        # Get optimal solution if available
        if dataset_type == 'real':
            sol_file = file_path.replace('input_data/real/prob', 'input_data/real/sol').replace('.dat', '.sln')
        else:
            sol_file = None
        
        optimal_value = read_optimal_solution(sol_file) if sol_file else None
        
        # Record start time
        start_time = time.time()
        
        # Solve using 2-opt
        solution, objective_value, total_iterations, restarts_used = two_opt_qap(
            dist_matrix, flow_matrix,
            max_iterations=max_iterations,
            time_budget_seconds=time_budget_seconds,
            neighborhood_size=neighborhood_size,
            random_restarts=random_restarts,
            seed=seed,
            verbose=verbose
        )
        
        # Calculate runtime
        runtime = time.time() - start_time
        
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
            'solution': solution,
            'total_iterations': total_iterations,
            'random_restarts': restarts_used,
            'neighborhood_size': neighborhood_size
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def setup_logging(dataset_type: str) -> str:
    """Set up logging configuration."""
    log_filename = f'log/qap_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{dataset_type}_2opt.log'
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


def main():
    parser = argparse.ArgumentParser(description='Solve QAP instances using 2-opt local search')
    parser.add_argument('--dataset', type=str, choices=['real', 'synthetic'], default='real',
                      help='Dataset type to solve (real or synthetic)')
    parser.add_argument('--max_iterations', type=int, default=10000,
                      help='Maximum iterations per restart (ignored if using time budget)')
    parser.add_argument('--time_budget', type=float, default=None,
                      help='Time budget in seconds (overrides max_iterations). Default: 2*n for large instances')
    parser.add_argument('--neighborhood_size', type=int, default=100,
                      help='Number of random swaps to try per iteration')
    parser.add_argument('--random_restarts', type=int, default=1,
                      help='Number of random restarts')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--large', action='store_true',
                      help='Use large instance settings (time budget mode)')
    parser.add_argument('--instance', type=str, default=None,
                      help='Specific instance file to solve (optional)')
    parser.add_argument('--verbose', action='store_true', default=True,
                      help='Enable verbose progress output (default: True)')
    parser.add_argument('--quiet', action='store_true',
                      help='Disable verbose progress output')
    
    args = parser.parse_args()
    
    # Handle verbose/quiet flags
    verbose = args.verbose and not args.quiet
    
    # Setup logging
    log_filename = setup_logging(args.dataset)
    
    logging.info(f"QAP 2-opt Solver")
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Random seed: {args.seed}")
    
    # Set directory based on dataset type
    if args.dataset == 'real':
        prob_dir = 'input_data/real/prob'
    else:  # synthetic
        prob_dir = 'input_data/synthetic'
    
    # Get instance files
    if args.instance:
        if os.path.exists(args.instance):
            instance_files = [os.path.basename(args.instance)]
            prob_dir = os.path.dirname(args.instance)
        else:
            instance_files = [args.instance]
    else:
        instance_files = [f for f in os.listdir(prob_dir) if f.endswith('.dat')]
        # Sort files by size (smaller instances first)
        instance_files.sort(key=lambda x: os.path.getsize(os.path.join(prob_dir, x)))
    
    for instance_file in instance_files:
        file_path = os.path.join(prob_dir, instance_file)
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing instance: {instance_file}")
        
        try:
            # Determine time budget for large instances
            time_budget = args.time_budget
            if args.large or time_budget is None:
                # Need to read the problem size first
                try:
                    with open(file_path, 'r') as f:
                        first_line = f.readline().strip()
                    n = int(first_line)
                    if time_budget is None:
                        time_budget = 2 * n  # Default time budget: 2*n seconds
                    logging.info(f"Using time budget: {time_budget}s for problem size {n}")
                except:
                    logging.warning(f"Could not determine problem size for auto time budget, using default")
                    time_budget = None
            
            # Solve the instance
            result = solve_qap_instance_2opt(
                file_path, args.dataset,
                max_iterations=args.max_iterations,
                time_budget_seconds=time_budget,
                neighborhood_size=args.neighborhood_size,
                random_restarts=args.random_restarts,
                seed=args.seed,
                verbose=verbose
            )
            
            if result['success']:
                logging.info(f"Objective value: {result['objective_value']:.0f}")
                if result['optimal_value'] is not None:
                    logging.info(f"Optimal value: {result['optimal_value']}")
                    logging.info(f"Gap: {result['gap']:.2f}%")
                else:
                    logging.info("No optimal value available for gap calculation")
                logging.info(f"Runtime: {result['runtime']:.2f} seconds")
                logging.info(f"Total iterations: {result['total_iterations']}")
                logging.info(f"Random restarts: {result['random_restarts']}")
                logging.info(f"Solution: {result['solution']}")
            else:
                logging.error(f"Failed to solve {instance_file}: {result['error']}")
                
        except Exception as e:
            logging.error(f"Error processing {instance_file}: {str(e)}")
    
    logging.info(f"\nResults saved to: {log_filename}")


if __name__ == "__main__":
    main()
