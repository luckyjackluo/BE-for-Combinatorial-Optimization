import json
import matplotlib.pyplot as plt

def read_loss_curves_from_json(json_path):
    """
    Reads a QAP loss curves JSON file and extracts the soft loss and hard loss (TL) curves
    for each problem and each run.
    Returns:
        results: dict of the form {problem_name: {run_index: (epochs, soft_loss, hard_loss)}}
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    results = {}
    for problem, runs in data.items():
        results[problem] = {}
        for run_idx, epoch_list in runs.items():
            epochs = [entry['epoch'] for entry in epoch_list]
            soft_loss = [entry['soft_loss'] for entry in epoch_list]
            hard_loss = [entry['hard_loss'] for entry in epoch_list]
            results[problem][int(run_idx)] = (epochs, soft_loss, hard_loss)
    return results

def read_gap_curves_from_json(json_path):
    """
    Reads a QAP gap curves JSON file and extracts the gap curves
    for each problem and each run.
    Returns:
        results: dict of the form {problem_name: {run_index: (epochs, gaps)}}
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    results = {}
    for problem, runs in data.items():
        results[problem] = {}
        for run_idx, epoch_list in runs.items():
            epochs = [entry['epoch'] for entry in epoch_list]
            gaps = [entry['gap'] for entry in epoch_list]
            results[problem][int(run_idx)] = (epochs, gaps)
    return results

def read_gap_curves_from_old_format(json_path):
    """
    Reads a QAP loss curves JSON file in the old format (with soft_loss and hard_loss)
    and converts it to gap curves format.
    This is for backward compatibility.
    Returns:
        results: dict of the form {problem_name: {run_index: (epochs, gaps)}}
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    results = {}
    for problem, runs in data.items():
        results[problem] = {}
        for run_idx, epoch_list in runs.items():
            epochs = [entry['epoch'] for entry in epoch_list]
            # Extract gaps from the old format
            gaps = [entry['gap'] for entry in epoch_list]
            results[problem][int(run_idx)] = (epochs, gaps)
    return results

def plot_loss_curves_for_run(results, problem_name, run_index, title=None):
    """
    Plots the loss curves for a specific problem and run.
    """
    if problem_name not in results or run_index not in results[problem_name]:
        print(f"No data for problem {problem_name}, run {run_index}")
        return
    epochs, soft_loss, hard_loss = results[problem_name][run_index]
    plt.figure(figsize=(10,6))
    plt.plot(epochs, soft_loss, label='Soft Loss')
    plt.plot(epochs, hard_loss, label='Hard Loss (TL)')
    # Add dashed red horizontal line for final converged best hard loss
    if len(hard_loss) > 0:
        best_hard_loss = min(hard_loss)
        plt.axhline(y=best_hard_loss, color='red', linestyle='--', label=f'Best Hard Loss: {best_hard_loss:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title or f'QAP Loss Curves: {problem_name}, Run {run_index}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_best_run_loss_curves_for_problem(results, problem_name, title=None, normalize=False):
    """
    Plots the loss curves for the best run (lowest final hard loss) for a given problem.
    If runs have different lengths, truncates to the shortest run.
    Adds a dashed red line for the best hard loss.
    
    Args:
        results: Dictionary containing loss curves data
        problem_name: Name of the problem to plot
        title: Optional title for the plot
        normalize: If True, normalize curves so initial loss = 1, otherwise use raw values
    """
    if problem_name not in results or len(results[problem_name]) < 1:
        print(f"No data for problem {problem_name}")
        return
    
    # Find the best run (lowest final hard loss)
    runs = results[problem_name]
    best_run_idx = None
    best_final_loss = float('inf')
    
    for run_idx in runs:
        epochs, soft_loss, hard_loss = runs[run_idx]
        if len(hard_loss) > 0:
            final_loss = hard_loss[-1]
            if final_loss < best_final_loss:
                best_final_loss = final_loss
                best_run_idx = run_idx
    
    if best_run_idx is None:
        print(f"No valid runs found for problem {problem_name}")
        return
    
    # Get the best run data
    epochs, soft_loss, hard_loss = runs[best_run_idx]
    
    if normalize:
        # Normalize using initial soft loss as reference (1.0)
        if len(soft_loss) > 0 and soft_loss[0] > 0:
            initial_soft_loss = soft_loss[0]
            soft_loss_normalized = [loss / initial_soft_loss for loss in soft_loss]
            hard_loss_normalized = [loss / initial_soft_loss for loss in hard_loss]
        else:
            soft_loss_normalized = soft_loss
            hard_loss_normalized = hard_loss
    else:
        soft_loss_normalized = soft_loss
        hard_loss_normalized = hard_loss
    plt.figure(figsize=(10,6))
    plt.plot(epochs, soft_loss_normalized, label='Soft Loss')
    plt.plot(epochs, hard_loss_normalized, label='Hard Loss (TL)')
    # Add dashed red horizontal line for best hard loss
    best_hard = min(hard_loss_normalized)
    plt.axhline(y=best_hard, color='red', linestyle='--', label=f'Best Hard Loss: {best_hard:.4f}')
    
    if normalize:
        # Add horizontal line at y=1 to show initial normalized value
        plt.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, alpha=0.7, label='Initial Soft Loss (Normalized)')
        plt.ylabel('Normalized Loss (Initial Soft Loss = 1.0)')
    else:
        plt.ylabel('Loss')
    
    plt.xlabel('Epoch')
    plt.title(title or f'Best Run QAP Loss Curves: {problem_name} (Run {best_run_idx})' + (' - Normalized' if normalize else ''))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_average_loss_curves_across_problems(results, problem_names=None, title=None, normalize=True):
    """
    Plots the average loss curves (soft and hard) across multiple problems using the best run from each problem.
    If normalize=True, normalizes each problem so that initial soft loss = 1, then scales all subsequent losses proportionally.
    If no problem_names provided, uses all problems in results.
    For each problem, selects the run with the lowest final hard loss.
    If runs have different lengths, truncates to the shortest run.
    Adds a dashed red line for the best average hard loss.
    
    Args:
        results: Dictionary containing loss curves data
        problem_names: List of problem names to include (None for all)
        title: Optional title for the plot
        normalize: If True, normalize curves so initial soft loss = 1, otherwise use raw values
    """
    if problem_names is None:
        problem_names = list(results.keys())
    
    if len(problem_names) < 2:
        print(f"Need at least 2 problems to plot average across problems, got {len(problem_names)}")
        return
    
    # Gather best runs from all specified problems
    best_runs_soft = []
    best_runs_hard = []
    total_problems = 0
    
    for problem_name in problem_names:
        if problem_name not in results:
            print(f"Warning: Problem {problem_name} not found in results, skipping")
            continue
            
        runs = results[problem_name]
        if len(runs) == 0:
            print(f"Warning: No runs found for problem {problem_name}, skipping")
            continue
        
        # Find the best run (lowest final hard loss) for this problem
        best_run_idx = None
        best_final_loss = float('inf')
        
        for run_idx in runs:
            epochs, soft_loss, hard_loss = runs[run_idx]
            if len(hard_loss) > 0:
                final_loss = hard_loss[-1]
                if final_loss < best_final_loss:
                    best_final_loss = final_loss
                    best_run_idx = run_idx
        
        if best_run_idx is None:
            print(f"Warning: No valid runs found for problem {problem_name}, skipping")
            continue
        
        # Get the best run data
        epochs, soft_loss, hard_loss = runs[best_run_idx]
        
        if normalize:
            # Normalize using initial soft loss as reference (1.0)
            if len(soft_loss) > 0 and soft_loss[0] > 0:
                initial_soft_loss = soft_loss[0]
                soft_loss_normalized = [loss / initial_soft_loss for loss in soft_loss]
                hard_loss_normalized = [loss / initial_soft_loss for loss in hard_loss]
            else:
                soft_loss_normalized = soft_loss
                hard_loss_normalized = hard_loss
        else:
            soft_loss_normalized = soft_loss
            hard_loss_normalized = hard_loss
        
        best_runs_soft.append(soft_loss_normalized)
        best_runs_hard.append(hard_loss_normalized)
        total_problems += 1
    
    if len(best_runs_soft) == 0:
        print("No valid runs found across specified problems")
        return
    
    # Find the minimum length across all best runs
    min_len = min(len(run) for run in best_runs_soft)
    
    # Truncate all best runs to the same length
    best_runs_soft = [run[:min_len] for run in best_runs_soft]
    best_runs_hard = [run[:min_len] for run in best_runs_hard]
    
    # Compute averages across best runs from all problems
    import numpy as np
    avg_soft = np.mean(np.array(best_runs_soft), axis=0)
    avg_hard = np.mean(np.array(best_runs_hard), axis=0)
    
    epochs = list(range(min_len))
    plt.figure(figsize=(12, 8))
    if normalize:
        plt.plot(epochs, avg_soft, label='Average Soft Loss (Normalized)', linewidth=2)
        plt.plot(epochs, avg_hard, label='Average Hard Loss (TL) (Normalized)', linewidth=2)
        
        # Add horizontal line at y=1 to show initial normalized value
        plt.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, alpha=0.7, label='Initial Soft Loss (Normalized)')
        plt.ylabel('Normalized Loss (Initial Soft Loss = 1.0)', fontsize=12)
    else:
        plt.plot(epochs, avg_soft, label='Average Soft Loss', linewidth=2)
        plt.plot(epochs, avg_hard, label='Average Hard Loss (TL)', linewidth=2)
        plt.ylabel('Loss', fontsize=12)
    
    # Add dashed red horizontal line for best average hard loss
    best_avg_hard = np.min(avg_hard)
    plt.axhline(y=best_avg_hard, color='red', linestyle='--', linewidth=2, 
                label=f'Best Avg Hard Loss: {best_avg_hard:.4f}')
    
    plt.xlabel('Epoch', fontsize=12)
    title_suffix = ' (Normalized)' if normalize else ''
    plt.title(title or f'Average QAP Loss Curves{title_suffix}: {len(problem_names)} problems, {total_problems} best runs', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    if normalize:
        print(f"Summary across {len(problem_names)} problems, {total_problems} best runs (Normalized):")
        print(f"  - Best average hard loss: {best_avg_hard:.4f} (relative to initial)")
        print(f"  - Final average soft loss: {avg_soft[-1]:.4f} (relative to initial)")
        print(f"  - Final average hard loss: {avg_hard[-1]:.4f} (relative to initial)")
        print(f"  - Average improvement: {(1.0 - best_avg_hard) * 100:.2f}%")
    else:
        print(f"Summary across {len(problem_names)} problems, {total_problems} best runs:")
        print(f"  - Best average hard loss: {best_avg_hard:.4f}")
        print(f"  - Final average soft loss: {avg_soft[-1]:.4f}")
        print(f"  - Final average hard loss: {avg_hard[-1]:.4f}")
    print(f"  - Problems included: {', '.join(problem_names)}")

def get_average_loss_curves_across_problems(results, problem_names=None, normalize=True):
    """
    Computes and returns the average loss curves (soft and hard) across multiple problems using the best run from each problem.
    If normalize=True, normalizes each problem so that initial soft loss = 1, then scales all subsequent losses proportionally.
    If no problem_names provided, uses all problems in results.
    For each problem, selects the run with the lowest final hard loss.
    If runs have different lengths, truncates to the shortest run.
    
    Args:
        results: Dictionary containing loss curves data
        problem_names: List of problem names to include (None for all)
        normalize: If True, normalize curves so initial soft loss = 1, otherwise use raw values
    
    Returns:
        dict: Contains 'epochs', 'avg_soft_loss', 'avg_hard_loss', 'problem_names', 'total_problems', 'best_avg_hard_loss'
    """
    if problem_names is None:
        problem_names = list(results.keys())
    
    if len(problem_names) < 1:
        print(f"Need at least 1 problem to compute average, got {len(problem_names)}")
        return None
    
    # Gather best runs from all specified problems
    best_runs_soft = []
    best_runs_hard = []
    total_problems = 0
    
    for problem_name in problem_names:
        if problem_name not in results:
            print(f"Warning: Problem {problem_name} not found in results, skipping")
            continue
            
        runs = results[problem_name]
        if len(runs) == 0:
            print(f"Warning: No runs found for problem {problem_name}, skipping")
            continue
        
        # Find the best run (lowest final hard loss) for this problem
        best_run_idx = None
        best_final_loss = float('inf')
        
        for run_idx in runs:
            epochs, soft_loss, hard_loss = runs[run_idx]
            if len(hard_loss) > 0:
                final_loss = hard_loss[-1]
                if final_loss < best_final_loss:
                    best_final_loss = final_loss
                    best_run_idx = run_idx
        
        if best_run_idx is None:
            print(f"Warning: No valid runs found for problem {problem_name}, skipping")
            continue
        
        # Get the best run data
        epochs, soft_loss, hard_loss = runs[best_run_idx]
        
        if normalize:
            # Normalize using initial soft loss as reference (1.0)
            if len(soft_loss) > 0 and soft_loss[0] > 0:
                initial_soft_loss = soft_loss[0]
                soft_loss_normalized = [loss / initial_soft_loss for loss in soft_loss]
                hard_loss_normalized = [loss / initial_soft_loss for loss in hard_loss]
            else:
                soft_loss_normalized = soft_loss
                hard_loss_normalized = hard_loss
        else:
            soft_loss_normalized = soft_loss
            hard_loss_normalized = hard_loss
        
        best_runs_soft.append(soft_loss_normalized)
        best_runs_hard.append(hard_loss_normalized)
        total_problems += 1
    
    if len(best_runs_soft) == 0:
        print("No valid runs found across specified problems")
        return None
    
    # Find the minimum length across all best runs
    min_len = min(len(run) for run in best_runs_soft)
    
    # Truncate all best runs to the same length
    best_runs_soft = [run[:min_len] for run in best_runs_soft]
    best_runs_hard = [run[:min_len] for run in best_runs_hard]
    
    # Compute averages across best runs from all problems
    import numpy as np
    avg_soft = np.mean(np.array(best_runs_soft), axis=0)
    avg_hard = np.mean(np.array(best_runs_hard), axis=0)
    
    epochs = list(range(min_len))
    best_avg_hard = np.min(avg_hard)
    
    return {
        'epochs': epochs,
        'avg_soft_loss': avg_soft.tolist(),
        'avg_hard_loss': avg_hard.tolist(),
        'problem_names': problem_names,
        'total_problems': total_problems,
        'best_avg_hard_loss': float(best_avg_hard),
        'normalized': normalize
    }

def plot_problem_comparison(results, problem_names, run_indices=None, title=None, normalize=False):
    """
    Plots comparison of specific runs from different problems on the same graph.
    If run_indices is None, uses the first run from each problem.
    If run_indices is a dict, uses specified run index for each problem.
    
    Args:
        results: Dictionary containing loss curves data
        problem_names: List of problem names to compare
        run_indices: Run indices to use (None for first run, int for same run, dict for specific runs)
        title: Optional title for the plot
        normalize: If True, normalize curves so initial loss = 1, otherwise use raw values
    """
    if len(problem_names) < 2:
        print(f"Need at least 2 problems to compare, got {len(problem_names)}")
        return
    
    if run_indices is None:
        run_indices = {problem: 0 for problem in problem_names}  # Use first run for each problem
    elif isinstance(run_indices, int):
        run_indices = {problem: run_indices for problem in problem_names}  # Use same run index for all
    
    plt.figure(figsize=(12, 8))
    
    for problem_name in problem_names:
        if problem_name not in results:
            print(f"Warning: Problem {problem_name} not found in results, skipping")
            continue
            
        run_idx = run_indices.get(problem_name, 0)
        if run_idx not in results[problem_name]:
            print(f"Warning: Run {run_idx} not found for problem {problem_name}, skipping")
            continue
            
        epochs, soft_loss, hard_loss = results[problem_name][run_idx]
        if normalize:
            # Normalize using initial soft loss as reference (1.0)
            if len(soft_loss) > 0 and soft_loss[0] > 0:
                initial_soft_loss = soft_loss[0]
                hard_loss_normalized = [loss / initial_soft_loss for loss in hard_loss]
            else:
                hard_loss_normalized = hard_loss
            plt.plot(epochs, hard_loss_normalized, label=f'{problem_name} (Run {run_idx})', linewidth=2)
        else:
            plt.plot(epochs, hard_loss, label=f'{problem_name} (Run {run_idx})', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    if normalize:
        plt.ylabel('Normalized Hard Loss (TL) (Initial Soft Loss = 1.0)', fontsize=12)
        title_suffix = ' (Normalized)'
    else:
        plt.ylabel('Hard Loss (TL)', fontsize=12)
        title_suffix = ''
    plt.title(title or f'QAP Problem Comparison: Hard Loss Curves{title_suffix}', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_gap_curves_for_run(results, problem_name, run_index, title=None):
    """
    Plots the gap curves for a specific problem and run.
    """
    if problem_name not in results or run_index not in results[problem_name]:
        print(f"No data for problem {problem_name}, run {run_index}")
        return
    epochs, gaps = results[problem_name][run_index]
    plt.figure(figsize=(10,6))
    plt.plot(epochs, gaps, label='Gap (%)', color='blue', linewidth=2)
    # Add dashed red horizontal line for final converged best gap
    if len(gaps) > 0:
        best_gap = min(gaps)
        plt.axhline(y=best_gap, color='red', linestyle='--', label=f'Best Gap: {best_gap:.2f}%')
    plt.xlabel('Epoch')
    plt.ylabel('Gap (%)')
    plt.title(title or f'QAP Gap Curves: {problem_name}, Run {run_index}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_best_run_gap_curves_for_problem(results, problem_name, title=None):
    """
    Plots the gap curves for the best run (lowest final gap) for a given problem.
    If runs have different lengths, truncates to the shortest run.
    Adds a dashed red line for the best gap.
    
    Args:
        results: Dictionary containing gap curves data
        problem_name: Name of the problem to plot
        title: Optional title for the plot
    """
    if problem_name not in results or len(results[problem_name]) < 1:
        print(f"No data for problem {problem_name}")
        return
    
    # Find the best run (lowest final gap)
    runs = results[problem_name]
    best_run_idx = None
    best_final_gap = float('inf')
    
    for run_idx in runs:
        epochs, gaps = runs[run_idx]
        if len(gaps) > 0:
            final_gap = gaps[-1]
            if final_gap < best_final_gap:
                best_final_gap = final_gap
                best_run_idx = run_idx
    
    if best_run_idx is None:
        print(f"No valid runs found for problem {problem_name}")
        return
    
    # Get the best run data
    epochs, gaps = runs[best_run_idx]
    
    plt.figure(figsize=(10,6))
    plt.plot(epochs, gaps, label='Gap (%)', color='blue', linewidth=2)
    # Add dashed red horizontal line for best gap
    best_gap = min(gaps)
    plt.axhline(y=best_gap, color='red', linestyle='--', label=f'Best Gap: {best_gap:.2f}%')
    
    plt.xlabel('Epoch')
    plt.ylabel('Gap (%)')
    plt.title(title or f'Best Run QAP Gap Curves: {problem_name} (Run {best_run_idx})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_average_gap_curves_across_problems(results, problem_names=None, title=None):
    """
    Plots the average gap curves across multiple problems using the best run from each problem.
    If no problem_names provided, uses all problems in results.
    For each problem, selects the run with the lowest final gap.
    If runs have different lengths, truncates to the shortest run.
    Adds a dashed red line for the best average gap.
    
    Args:
        results: Dictionary containing gap curves data
        problem_names: List of problem names to include (None for all)
        title: Optional title for the plot
    """
    if problem_names is None:
        problem_names = list(results.keys())
    
    if len(problem_names) < 2:
        print(f"Need at least 2 problems to plot average across problems, got {len(problem_names)}")
        return
    
    # Gather best runs from all specified problems
    best_runs_gaps = []
    total_problems = 0
    
    for problem_name in problem_names:
        if problem_name not in results:
            print(f"Warning: Problem {problem_name} not found in results, skipping")
            continue
            
        runs = results[problem_name]
        if len(runs) == 0:
            print(f"Warning: No runs found for problem {problem_name}, skipping")
            continue
        
        # Find the best run (lowest final gap) for this problem
        best_run_idx = None
        best_final_gap = float('inf')
        
        for run_idx in runs:
            epochs, gaps = runs[run_idx]
            if len(gaps) > 0:
                final_gap = gaps[-1]
                if final_gap < best_final_gap:
                    best_final_gap = final_gap
                    best_run_idx = run_idx
        
        if best_run_idx is None:
            print(f"Warning: No valid runs found for problem {problem_name}, skipping")
            continue
        
        # Get the best run data
        epochs, gaps = runs[best_run_idx]
        best_runs_gaps.append(gaps)
        total_problems += 1
    
    if len(best_runs_gaps) == 0:
        print("No valid runs found across specified problems")
        return
    
    # Find the minimum length across all best runs
    min_len = min(len(run) for run in best_runs_gaps)
    
    # Truncate all best runs to the same length
    best_runs_gaps = [run[:min_len] for run in best_runs_gaps]
    
    # Compute averages across best runs from all problems
    import numpy as np
    avg_gaps = np.mean(np.array(best_runs_gaps), axis=0)
    
    epochs = list(range(min_len))
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, avg_gaps, label='Average Gap (%)', color='blue', linewidth=2)
    
    # Add dashed red horizontal line for best average gap
    best_avg_gap = np.min(avg_gaps)
    plt.axhline(y=best_avg_gap, color='red', linestyle='--', linewidth=2, 
                label=f'Best Avg Gap: {best_avg_gap:.2f}%')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Gap (%)', fontsize=12)
    plt.title(title or f'Average QAP Gap Curves: {len(problem_names)} problems, {total_problems} best runs', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"Summary across {len(problem_names)} problems, {total_problems} best runs:")
    print(f"  - Best average gap: {best_avg_gap:.2f}%")
    print(f"  - Final average gap: {avg_gaps[-1]:.2f}%")
    print(f"  - Average improvement: {(avg_gaps[0] - best_avg_gap):.2f}%")
    print(f"  - Problems included: {', '.join(problem_names)}")

def plot_problem_gap_comparison(results, problem_names, run_indices=None, title=None):
    """
    Plots comparison of specific runs from different problems on the same graph.
    If run_indices is None, uses the first run from each problem.
    If run_indices is a dict, uses specified run index for each problem.
    
    Args:
        results: Dictionary containing gap curves data
        problem_names: List of problem names to compare
        run_indices: Run indices to use (None for first run, int for same run, dict for specific runs)
        title: Optional title for the plot
    """
    if len(problem_names) < 2:
        print(f"Need at least 2 problems to compare, got {len(problem_names)}")
        return
    
    if run_indices is None:
        run_indices = {problem: 0 for problem in problem_names}  # Use first run for each problem
    elif isinstance(run_indices, int):
        run_indices = {problem: run_indices for problem in problem_names}  # Use same run index for all
    
    plt.figure(figsize=(12, 8))
    
    for problem_name in problem_names:
        if problem_name not in results:
            print(f"Warning: Problem {problem_name} not found in results, skipping")
            continue
            
        run_idx = run_indices.get(problem_name, 0)
        if run_idx not in results[problem_name]:
            print(f"Warning: Run {run_idx} not found for problem {problem_name}, skipping")
            continue
            
        epochs, gaps = results[problem_name][run_idx]
        plt.plot(epochs, gaps, label=f'{problem_name} (Run {run_idx})', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Gap (%)', fontsize=12)
    plt.title(title or f'QAP Problem Comparison: Gap Curves', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Example usage:
# results = read_loss_curves_from_json('log/loss_curves_gd+k+5_real_0_20240503_221500.json')
# plot_loss_curves_for_run(results, 'qap_n20_d1.00_id0.dat', 1) 
# plot_best_run_loss_curves_for_problem(results, 'qap_n20_d1.00_id0.dat', normalize=True)
# plot_average_loss_curves_across_problems(results, ['qap_n20_d1.00_id0.dat', 'qap_n20_d1.00_id1.dat'], normalize=True)
# plot_problem_comparison(results, ['qap_n20_d1.00_id0.dat', 'qap_n20_d1.00_id1.dat'], normalize=False)
# 
# Get average loss curves data without plotting:
# avg_data = get_average_loss_curves_across_problems(results, ['qap_n20_d1.00_id0.dat', 'qap_n20_d1.00_id1.dat'], normalize=True)
# epochs = avg_data['epochs']
# avg_soft = avg_data['avg_soft_loss']
# avg_hard = avg_data['avg_hard_loss']

# New gap curve functions:
# gap_results = read_gap_curves_from_json('log/gap_curves_gd+k+5_real_0_20240503_221500.json')
# plot_gap_curves_for_run(gap_results, 'qap_n20_d1.00_id0.dat', 1)
# plot_best_run_gap_curves_for_problem(gap_results, 'qap_n20_d1.00_id0.dat')
# plot_average_gap_curves_across_problems(gap_results, ['qap_n20_d1.00_id0.dat', 'qap_n20_d1.00_id1.dat'])
# plot_problem_gap_comparison(gap_results, ['qap_n20_d1.00_id0.dat', 'qap_n20_d1.00_id1.dat'])