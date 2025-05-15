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

from torch import Tensor
from torch.utils.data import DataLoader
from numpy.random import default_rng
from tqdm import tqdm
from collections import OrderedDict
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import maximum_bipartite_matching

def compute_qap_cost_torch(A, B, P):
    """
    A: (N,N) torch tensor
    B: (N,N) torch tensor
    P: (N,N) torch permutation matrix
    """
    # Ensure all tensors are float32
    A = A.to(dtype=torch.float32)
    B = B.to(dtype=torch.float32)
    P = P.to(dtype=torch.float32)
    PBPT = P @ B @ P.T   # P B P^T
    return torch.sum(A * PBPT)

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

def setup_logging(dataset_type, lr, alg, device_idx):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"train_be/results_{lr}_qap_{alg}_{dataset_type}_{device_idx}_{timestamp}.log"
    os.makedirs("train_be", exist_ok=True)
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
    
    # To store all permutations and their costs
    all_P = []
    all_tl = []
    
    # Pre-allocate P matrix
    P = torch.zeros(n, n, device=device)
    
    for idx in range(k):
        # Compute assignment matrix in one go
        A_to_use = torch.where(new_W > 0, perms, fill).detach().cpu().numpy()
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(A_to_use, maximize=True)
        
        # Get threshold and update sum in one operation
        thresh = torch.amin(new_W[row_ind, col_ind])
        sum_thresh += thresh
        
        # Create permutation matrix efficiently
        P.zero_()  # Reset P matrix
        P[row_ind, col_ind] = 1
        
        # Store this permutation
        all_P.append(P.clone())
        
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
    
    return min_tl, total_loss/sum_thresh, min_P, idx, sum_thresh.item(), all_P, all_tl

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
    parser.add_argument('--device_idx', type=int, required=True,
                      help='Device index to use')
    parser.add_argument('--visualize', action='store_true',
                      help='Enable visualization of permutation distributions')
    args = parser.parse_args()

    S = "constant"  # Only use constant strategy
    alg = args.alg
    lr = args.lr
    device_idx = args.device_idx
    update_best_perms = args.update_best_perms == 1
    print(device_idx)
    device = torch.device("cpu")
    print(sys.argv)
    T = 500
    gen = True
    input_lst = []

    # Set directory based on dataset type
    if args.dataset == 'real':
        qap_dir = "qap/prob"
        sol_dir = "qap/sol"
    else:  # synthetic
        qap_dir = "qap/input_data/synthetic"
        sol_dir = None  # No solution files for synthetic instances

    # Load QAP instances
    all_files = os.listdir(qap_dir)
    dataset = []
    optimal_values = {}
    found = False

    # For synthetic data, filter for specific instances
    target_ids = [352, 124, 384, 49, 158]
    
    for fp in all_files:
        if fp.endswith('.dat'):
            # For synthetic data, only process specific instance IDs
            if args.dataset == 'synthetic':
                # Extract the ID from the filename (assuming format like instance_ID.dat)
                try:
                    file_id = int(fp.split('.d')[0].split('_')[-1].split('d')[-1])
                    if file_id not in target_ids:
                        continue  # Skip this file if not in target_ids
                except (ValueError, IndexError):
                    # If there's an issue parsing the ID, try another format
                    try:
                        file_id = int(fp.split('.')[0])
                        if file_id not in target_ids:
                            continue
                    except (ValueError, IndexError):
                        # If we can't parse the ID at all, skip this file
                        continue
            
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

    # Setup logging
    log_filename = setup_logging(args.dataset, lr, alg, device_idx)

    with open(log_filename, 'w') as f:
        f.write(f"Training Log - {datetime.datetime.now()}\n")
        f.write(f"Parameters: dataset={args.dataset}, lr={lr}, alg={alg}, device_idx={device_idx}\n\n")

    alg_lst = alg.split("+")
    if alg_lst[1] == "k":
        setting = [alg_lst[1], int(alg_lst[2])] 
    elif alg_lst[1] == "p":
        setting = [alg_lst[1], float(alg_lst[2])]
    print(alg_lst, setting)

    # Initialize results storage
    num_runs = 3
    best_results = {}  # Store best results for each problem

    for j_idx, (fp, n, A, B) in enumerate(dataset):
        print(f"\nProcessing {fp}")
        best_tl_overall = float('inf')
        best_run_idx = 0
        
        with open(log_filename, 'a') as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"Processing {fp}\n")
            f.write(f"{'='*50}\n")
        
        # Create visualization directory if needed
        if args.visualize:
            viz_dir = f"qap_viz/{fp.replace('.dat', '')}"
            os.makedirs(viz_dir, exist_ok=True)
        
        for run_idx in range(num_runs):
            setting[1] = run_idx*5 + 5
            print(f"\nRun {run_idx + 1}/{num_runs}")
            num_terms = n
            
            # Start timing this run
            run_start_time = time.time()
            
            # Initialize with constant permutation matrix
            perms = ipf(torch.rand(num_terms, num_terms)/num_terms, 5, 1, 1)
            
            model = MatrixModel(num_terms, alg="gd").to(device) 
            state_dict = model.state_dict()
            # Initialize with better weights
            weight = torch.rand(num_terms, num_terms, device=device)
            weight = torch.abs(weight)
            weight = weight / weight.sum(dim=1, keepdim=True)  # Row normalization
            weight = weight / weight.sum(dim=0, keepdim=True)  # Column normalization
            state_dict['mat.weight'] = ipf(weight, 5, 1, 1)  # More IPF iterations
            
            model.load_state_dict(state_dict)
            hashmap = OrderedDict()
            patience = 2000
            best_tl = float('inf')
            best_perms = perms.clone()
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
                    'all_perms': [],  # New: store all permutations for each epoch
                    'all_tls': []     # New: store all permutation costs for each epoch
                }

            for idx in range(T):
                W = model.forward() 
                tl, loss, perms, num_P, sum_thresh, all_P, all_tl = cont_Birkhoff_SFE(W, setting[1], A, B, best_perms, setting, device=device)
                loss.backward()
                
                # Store visualization data
                if args.visualize and (idx % 10 == 0 or idx < 10):
                    viz_data['epochs'].append(idx)
                    viz_data['W_heatmaps'].append(W.detach().cpu().numpy())
                    viz_data['perm_heatmaps'].append(perms.detach().cpu().numpy())
                    viz_data['losses'].append(loss.item())
                    viz_data['tls'].append(tl.item())
                    viz_data['best_tls'].append(best_tl.item() if idx > 0 else tl.item())
                    viz_data['all_perms'].append(all_P)
                    viz_data['all_tls'].append(all_tl)
                 
                if tl < best_tl:
                    best_tl = tl
                    if update_best_perms:
                        best_perms = perms
                    weight = torch.rand(num_terms, num_terms, device=device)
                    weight = torch.abs(weight)
                    weight = weight / weight.sum(dim=1, keepdim=True)
                    weight = weight / weight.sum(dim=0, keepdim=True)
                    state_dict['mat.weight'] = weight
                    model.load_state_dict(state_dict)
                    pt_en = 0
                    setting[1] = int(alg_lst[2])
                    patience = 2000

                else:
                    patience -= 1
                    loss_gap = (abs(loss - tl).item())/(tl.item() + 0.001)
                    if loss_gap <= 0.005:
                        pt_en += 0.01
                    elif loss_gap >= 0.1:
                        pt_en -= 0.01

                    if pt_en >= 1 and setting[1] <= num_terms:
                        setting[1] = int(setting[1] * 1.3) + 1
                        weight = torch.rand(num_terms, num_terms, device=device)
                        weight = torch.abs(weight)
                        weight = weight / weight.sum(dim=1, keepdim=True)
                        weight = weight / weight.sum(dim=0, keepdim=True)
                        state_dict['mat.weight'] = ipf(weight, 10, 1, 1)
                        model.load_state_dict(state_dict)
                        pt_en = 0
   
                    if pt_en <= -1 and setting[1] > int(alg_lst[2]):
                        setting[1] = np.min([int(setting[1] / 1.1), int(alg_lst[2])])
                        weight = torch.rand(num_terms, num_terms, device=device)
                        weight = torch.abs(weight)
                        weight = weight / weight.sum(dim=1, keepdim=True)
                        weight = weight / weight.sum(dim=0, keepdim=True)
                        state_dict['mat.weight'] = ipf(weight, 10, 1, 1)
                        model.load_state_dict(state_dict)
                        pt_en = 0                

                if idx // 10 > 0 and idx % 10 == 0:
                    optimal_value = optimal_values.get(fp, None)
                    if optimal_value == 0:
                        gap = 0
                    else:
                        gap = ((tl.item() - optimal_value) / optimal_value * 100) if optimal_value is not None else float('inf')
                    print(round(loss.item(), 4), round(tl.item(), 4), round(best_tl.item(), 4), 
                          f"gap: {round(gap, 2)}%" if optimal_value is not None else "gap: N/A",
                          num_P, round(sum_thresh, 4), patience, pt_en)
                
                if "pgd" in alg:
                    for param in model.parameters():
                        grad = param.grad.data
                        grad = grad - grad.mean(dim=1, keepdim=True)
                        grad = grad - grad.mean(dim=0, keepdim=True)
                        row, col = linear_sum_assignment(grad.cpu())
                        P = torch.zeros_like(W)
                        for i, j in zip(row, col):
                            P[i, j] = 1
                        grad_norm = torch.norm(grad)
                        if grad_norm > 0:
                            step_size = lr / (1 + 2 * grad_norm)
                        else:
                            step_size = lr
                        if not hasattr(param, 'momentum_buffer'):
                            param.momentum_buffer = torch.zeros_like(param.data)
                        param.momentum_buffer = 0.9 * param.momentum_buffer + step_size * (P - param.data)
                        param.data = param.data + param.momentum_buffer
                        torch.nn.Module.zero_grad(model)
                else:
                    optimizer.step()
                    optimizer.zero_grad()

                if patience <= 0:
                    run_time = time.time() - run_start_time
                    
                    # Create visualizations at the end of the run
                    if args.visualize:
                        create_visualizations(viz_data, fp, run_idx, viz_dir)
                    break
            
            # Create visualizations if we reached the end of training
            if args.visualize and patience > 0:
                create_visualizations(viz_data, fp, run_idx, viz_dir)
        
        # Update best overall result
        if best_tl < best_tl_overall:
            best_tl_overall = best_tl
            best_run_idx = run_idx
    
        # Write summary for this problem
        optimal_value = optimal_values.get(fp, None)
        if best_tl_overall.item() == optimal_value:
            gap = 0
        else:
            if optimal_value == 0:
                gap = 0
            else:
                gap = ((best_tl_overall.item() - optimal_value) / optimal_value * 100) if optimal_value is not None else float('inf')
        
        with open(log_filename, 'a') as f:
            f.write(f"\nSummary for {fp}:\n")
            f.write(f"Best TL: {best_tl_overall}\n")
            f.write(f"Best run: {best_run_idx + 1}\n")
            if optimal_value is not None:
                f.write(f"Final gap: {gap:.2f}%\n")
            f.write(f"{'='*50}\n")

    print(f"\nResults have been saved to {log_filename}")

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
        
        # Get permutation data
        all_perms = viz_data['all_perms'][i]
        all_tls = viz_data['all_tls'][i]
        
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
        ax_sum.plot(epochs_to_plot, viz_data['tls'][:i+1], label='Best TL', marker='x')
        ax_sum.set_xlabel('Epoch')
        ax_sum.set_ylabel('Value')
        ax_sum.legend()
        ax_sum.grid(True)
        
        # Create bar plot of permutation costs in sequential order (middle left)
        ax_cost_bar = fig.add_subplot(gs[1, 0])
        x_pos = range(len(all_tls))
        ax_cost_bar.bar(x_pos, all_tls)
        ax_cost_bar.set_title(f'Permutation Costs (Sequential Order)')
        ax_cost_bar.set_xlabel('Permutation Index')
        ax_cost_bar.set_ylabel('QAP Cost')
        
        # Create distribution plot of permutation costs (middle right)
        ax_cost_dist = fig.add_subplot(gs[1, 1])
        ax_cost_dist.hist(all_tls, bins=min(20, len(all_tls)//2 + 1), alpha=0.7)
        ax_cost_dist.set_title(f'Distribution of Permutation Costs')
        ax_cost_dist.set_xlabel('QAP Cost')
        ax_cost_dist.set_ylabel('Frequency')
        
        # Flatten all permutation matrices into rows for visualization
        # Create a sequential view of all permutation values (bottom left)
        if len(all_perms) > 0:
            # Select a subset if there are too many permutations
            max_perms_to_display = min(len(all_perms), 10)
            display_indices = np.linspace(0, len(all_perms)-1, max_perms_to_display, dtype=int)
            
            # Extract selected permutations and their costs
            selected_perms = [all_perms[idx].cpu().numpy() for idx in display_indices]
            selected_costs = [all_tls[idx] for idx in display_indices]
            
            # Create a bar plot showing permutation values in sequence
            ax_perm_seq = fig.add_subplot(gs[2, 0])
            
            # For each permutation, extract values in row-major order and plot as bars
            bar_width = 0.8 / max_perms_to_display
            for p_idx, (perm, cost) in enumerate(zip(selected_perms, selected_costs)):
                # Flatten the permutation matrix (row-major order)
                flat_perm = perm.flatten()
                x_pos = np.arange(len(flat_perm))
                ax_perm_seq.bar(x_pos + p_idx * bar_width - 0.4 + bar_width/2, 
                               flat_perm, 
                               width=bar_width, 
                               alpha=0.7,
                               label=f'Perm {display_indices[p_idx]} (Cost: {cost:.2f})')
            
            ax_perm_seq.set_title(f'Permutation Values (Sequential View)')
            ax_perm_seq.set_xlabel('Position (flattened matrix index)')
            ax_perm_seq.set_ylabel('Value')
            ax_perm_seq.legend(loc='upper right', fontsize='small')
            
            # Create a distribution plot of all permutation values (bottom right)
            ax_perm_dist = fig.add_subplot(gs[2, 1])
            
            # Combine all permutation values
            all_values = []
            for perm in all_perms:
                all_values.extend(perm.cpu().numpy().flatten())
            
            ax_perm_dist.hist(all_values, bins=20, alpha=0.7)
            ax_perm_dist.set_title(f'Distribution of All Permutation Values')
            ax_perm_dist.set_xlabel('Permutation Value')
            ax_perm_dist.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/run{run_idx}_epoch{epoch}.png")
        plt.close(fig)
    
    # Create a summary plot showing optimization progress
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot loss curve
    ax1.plot(viz_data['epochs'], viz_data['losses'], label='Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Loss vs Epoch for {problem_name} (Run {run_idx})')
    ax1.grid(True)
    
    # Plot TL
    ax2.plot(viz_data['epochs'], viz_data['tls'], label='TL')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('QAP Cost')
    ax2.set_title(f'QAP Cost vs Epoch for {problem_name} (Run {run_idx})')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/run{run_idx}_summary.png")
    plt.close(fig)
    
    # Create a sequential visualization of permutation costs across epochs
    if len(selected_indices) > 0:
        # Create figure for sequential visualization
        fig, axes = plt.subplots(len(selected_indices), 1, figsize=(14, 4*len(selected_indices)))
        if len(selected_indices) == 1:
            axes = [axes]  # Make sure axes is always a list
        
        # For each selected epoch, create a bar chart of permutation costs
        for ax_idx, i in enumerate(selected_indices):
            epoch = viz_data['epochs'][i]
            all_tls = viz_data['all_tls'][i]
            
            # Create bar plot
            axes[ax_idx].bar(range(len(all_tls)), all_tls)
            axes[ax_idx].set_title(f'Epoch {epoch}: Permutation Costs (Sequential Order)')
            axes[ax_idx].set_xlabel('Permutation Index')
            axes[ax_idx].set_ylabel('QAP Cost')
            axes[ax_idx].grid(True, axis='y')
            
            # Add a horizontal line for the minimum cost
            min_cost = min(all_tls)
            axes[ax_idx].axhline(y=min_cost, color='red', linestyle='--', 
                               label=f'Min Cost: {min_cost:.2f}')
            axes[ax_idx].legend()
        
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/run{run_idx}_sequential_costs.png")
        plt.close(fig)
    
    # Create a visualization showing the distribution of costs across permutations
    if num_epochs > 0:
        # Select a few epochs to visualize
        selected_epochs = selected_indices
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for i in selected_epochs:
            epoch = viz_data['epochs'][i]
            costs = viz_data['all_tls'][i]
            
            # Create a violin plot for the distribution of costs
            positions = [epoch] * len(costs)
            vp = ax.violinplot([costs], positions=[epoch], widths=5, showmeans=True)
            
            # Plot individual costs as points
            ax.scatter([epoch] * len(costs), costs, alpha=0.5, s=20)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('QAP Cost')
        ax.set_title(f'Distribution of QAP Costs Across Permutations ({problem_name}, Run {run_idx})')
        ax.grid(True)
        
        plt.savefig(f"{viz_dir}/run{run_idx}_cost_distributions.png")
        plt.close(fig)
        
        # Create a heatmap visualization of costs across all permutations and epochs
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Find the maximum number of permutations across all epochs
        max_perms = max(len(viz_data['all_tls'][i]) for i in selected_indices)
        
        # Create a 2D array for the heatmap
        heatmap_data = np.zeros((len(selected_indices), max_perms))
        heatmap_data.fill(np.nan)  # Fill with NaN for permutations that don't exist
        
        # Fill in the costs
        for i, epoch_idx in enumerate(selected_indices):
            costs = viz_data['all_tls'][epoch_idx]
            heatmap_data[i, :len(costs)] = costs
        
        # Create the heatmap
        im = ax.imshow(heatmap_data, aspect='auto', cmap='viridis')
        
        # Set ticks and labels
        epoch_labels = [viz_data['epochs'][i] for i in selected_indices]
        ax.set_yticks(range(len(selected_indices)))
        ax.set_yticklabels([f'Epoch {e}' for e in epoch_labels])
        ax.set_xlabel('Permutation Index')
        ax.set_ylabel('Epoch')
        ax.set_title(f'QAP Costs Across Permutations and Epochs ({problem_name}, Run {run_idx})')
        
        # Add a colorbar
        plt.colorbar(im, ax=ax, label='QAP Cost')
        
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/run{run_idx}_cost_heatmap.png")
        plt.close(fig)

if __name__ == "__main__":
    main()
