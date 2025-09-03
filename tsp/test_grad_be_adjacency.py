import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from IPF import *
import matplotlib.pyplot as plt
import networkx as nx
import scipy

from pathlib import Path
import re

import time
import logging
import os
import sys
import warnings
import itertools as it
from datetime import datetime
def append_path(s):
    if s in sys.path:
        return
    sys.path.insert(0, s)

append_path("..")

import seaborn as sns
from torch import Tensor
from torch.utils.data import DataLoader
from numpy.random import default_rng

from tqdm import tqdm

# Import the new adjacency-based Birkhoff TSP functions
from Birkhoff_TSP_adjacency import *
from collections import OrderedDict
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import maximum_bipartite_matching

def load_heatmaps_ordered(num_terms, max_instances):
    """Load heatmaps in order based on data indices."""
    heatmap_dir = Path(f"test_data/{num_terms}/numpy_heatmap")
    print(f"Loading heatmaps from {heatmap_dir}")
    
    heatmaps = []
    loaded_count = 0
    
    for data_idx in range(max_instances):
        heatmap_file = heatmap_dir / f"test-heatmap-{data_idx}.npy"
        
        if heatmap_file.exists():
            try:
                heatmap = np.load(str(heatmap_file))
                if heatmap.shape == (1, num_terms, num_terms):
                    heatmap_tensor = torch.tensor(heatmap[0], dtype=torch.float32)
                elif heatmap.shape == (num_terms, num_terms):
                    heatmap_tensor = torch.tensor(heatmap, dtype=torch.float32)
                else:
                    print(f"Warning: Unexpected shape {heatmap.shape} for {heatmap_file.name}, using None")
                    heatmap_tensor = None
                
                heatmaps.append(heatmap_tensor)
                if heatmap_tensor is not None:
                    loaded_count += 1
                    
            except Exception as e:
                print(f"Error loading {heatmap_file.name}: {e}")
                heatmaps.append(None)
        else:
            heatmaps.append(None)
    
    print(f"Successfully loaded {loaded_count} heatmaps out of {max_instances} instances")
    return heatmaps

def convert_perm_to_adjacency(perm_matrix):
    """Convert permutation matrix to adjacency matrix representation."""
    # For TSP, we can interpret the permutation matrix as edge probabilities
    # This creates an adjacency matrix where entry (i,j) represents the probability
    # of an edge between nodes i and j
    return perm_matrix.clone()

S = str(sys.argv[2]) 
alg = str(sys.argv[1])
lr = float(sys.argv[3])
target_num_terms = int(sys.argv[4])
mode = str(sys.argv[5])  # New mode argument: "train" or "eval"
device_idx = int(sys.argv[6])  # Shifted from sys.argv[-1]

# Validate mode argument
if mode not in ["train", "eval"]:
    raise ValueError(f"Mode must be 'train' or 'eval', got: {mode}")

# Setup logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"logs/adjacency_{alg}_{S}_{lr}_{target_num_terms}_{mode}_{device_idx}_{timestamp}.log"
os.makedirs("logs", exist_ok=True)

# Configure logging to write to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)

def log_print(*args, **kwargs):
    """Print function that also logs to file"""
    message = ' '.join(str(arg) for arg in args)
    logging.info(message)

# Replace print with log_print for the rest of the script
print = log_print

print(f"Device index: {device_idx}")
print(f"Mode: {mode}")
#device = torch.device(f"cuda:{device_idx}")
device = torch.device("cpu")
print(f"Arguments: {sys.argv}")
print(f"Log file: {log_filename}")
T = 5000
num_terms = target_num_terms
target_num_input = 1000
gen = True
input_lst = []
data_lst = torch.load(f"input_data/data_lst_{target_num_terms}.pt")
cost_lst = torch.load(f"input_data/cost_lst_{target_num_terms}.pt")
dist_lst = torch.load(f"input_data/dist_lst_{target_num_terms}.pt")

change_lst = []
train_curve = [[0 for idx in range(T//10)] for i in range(target_num_input)]
hard_train_curve = [[0 for idx in range(T//10)] for i in range(target_num_input)]
hashmap_size = []
assert len(train_curve) == target_num_input

alg_lst = alg.split("+")
if alg_lst[1] == "k":
    setting = [alg_lst[1], int(alg_lst[2])] 
elif alg_lst[1] == "p":
    setting = [alg_lst[1], float(alg_lst[2])]
print(alg_lst, setting)

idx = 0
print(f"Starting data preparation - target_num_input: {target_num_input}, num_terms: {num_terms}")


print("Loading existing input list...")
input_lst = torch.load(f"input_data/input_lst_{num_terms}_adjacency.pt")
print(f"Loaded input_lst with {len(input_lst)} instances")

# Load heatmaps in order
print("Loading heatmaps for warm start...")
heatmaps = load_heatmaps_ordered(num_terms, len(input_lst))

# Update target_num_input to match actual data
target_num_input = len(input_lst)
train_curve = [[0 for idx in range(T//10)] for i in range(target_num_input)]
hard_train_curve = [[0 for idx in range(T//10)] for i in range(target_num_input)]

class MatrixModel(nn.Module):
    def __init__(self, num_terms):
        super().__init__()
        self.num_terms = num_terms
        self.mat = torch.nn.Linear(num_terms, num_terms, bias=False)
    def forward(self):
        W = torch.abs(self.mat.weight) 
        W = ipf(W, 10, 1,1)
        return W

class AdjacencyModel(nn.Module):
    """Model that generates adjacency matrices for edge heatmaps."""
    def __init__(self, num_terms):
        super().__init__()
        self.num_terms = num_terms
        self.mat = torch.nn.Linear(num_terms, num_terms, bias=False)
    def forward(self):
        W = torch.sigmoid(self.mat.weight)  # Use sigmoid for adjacency probabilities
        # Make symmetric for undirected graph
        W = (W + W.T) / 2
        # Remove self-loops
        W = W * (1 - torch.eye(self.num_terms))
        return W

alg_lst = alg.split("+")
if alg_lst[1] == "k":
    setting = [alg_lst[1], int(alg_lst[2])] 
elif alg_lst[1] == "p":
    setting = [alg_lst[1], float(alg_lst[2])]
print(alg_lst, setting)

assert len(train_curve) == target_num_input 
assert len(input_lst) == target_num_input
print(f"Starting optimization with algorithm: {alg}")
print(f"Total instances to process: {len(input_lst)}")

# Determine which instances to process based on mode
if mode == "eval":
    instances_to_process = [0]  # Only process first instance
    print(f"Eval mode: Processing only first instance (data_idx: 0)")
else:  # train mode
    instances_to_process = list(range(len(input_lst)))
    print(f"Train mode: Processing all {len(input_lst)} instances")

# Storage for eval mode adjacency matrices
eval_adjacency_matrices = {}

for j_idx in tqdm.tqdm(instances_to_process):
    points, cost, D = input_lst[j_idx]    
    
    print(f"\n=== Processing instance {j_idx+1}/{len(instances_to_process)} (data_idx: {j_idx}) ===")
    print(f"Target cost: {cost:.4f}")
    
    # Initialize best_perm with heatmap if available
    if j_idx < len(heatmaps) and heatmaps[j_idx] is not None:
        best_perm = heatmaps[j_idx].clone()
        print(f"Initialized best_perm with heatmap for data index {j_idx}")
    else:
        print(f"No heatmap available for data index {j_idx}, skipping this instance")
        continue
    
    # Initialize W as doubly stochastic matrix based on S parameter
    model_W = MatrixModel(num_terms).to(device) 
    state_dict = model_W.state_dict()
    if S == "constant":
        # Use barycenter-style uniform doubly stochastic matrix
        state_dict['mat.weight'] = ipf(torch.ones(num_terms, num_terms)/num_terms, 5, 1, 1)
    else:  # S == "random"
        # Use random initialization
        weight = torch.rand(num_terms, num_terms, device=device)
        weight = torch.abs(weight)
        weight = weight / weight.sum(dim=1, keepdim=True)  # Row normalization
        weight = weight / weight.sum(dim=0, keepdim=True)  # Column normalization
        state_dict['mat.weight'] = ipf(weight, 5, 1, 1)  # IPF iterations for doubly stochastic
    model_W.load_state_dict(state_dict)
    optimizer_W = torch.optim.Adam(model_W.parameters(), lr=0.01)
    
    hashmap = OrderedDict()
    patience = 4000
    best_tl = num_terms**2 * 100  # Large initial value
    pt_en = 0

    for idx in range(T):
        total_perms_loss = 0
        W = model_W()
        
        # Save best_perm as adjacency matrix for eval mode
        if mode == "eval" and j_idx == 0:
            if idx == 1:  # First epoch
                eval_adjacency_matrices['first_epoch'] = convert_perm_to_adjacency(returned_best_perm).detach().cpu().clone()
                print(f"Saved first epoch best_perm as adjacency matrix")
            elif idx == T - 1:  # Last epoch
                eval_adjacency_matrices['last_epoch'] = convert_perm_to_adjacency(best_perm).detach().cpu().clone()
                print(f"Saved last epoch best_perm as adjacency matrix")
        
        # Use the adjacency version of Birkhoff SFE with pre-computed distance matrix
        W_loss, perms_loss, min_tl, sum_thresh, min_tour, returned_best_perm = cont_Birkhoff_SFE_adjacency(
            W, 5, points, best_perm, hashmap, setting, alg, distance_matrix=D, device=device
        )
        W_loss.backward()
        total_perms_loss += perms_loss
        optimizer_W.step()
        optimizer_W.zero_grad()
        
        if min_tl < best_tl:
            best_tl = min_tl
            if returned_best_perm is not None:
                best_perm = returned_best_perm  # Update best permutation from function return
            pt_en = 0
        else:
            patience -= 1

        # Note: No backward step on perms model, similar to dynamic_k version

        opt_gap = abs(min_tl.item() - cost)
        
        if idx // 10 > 0 and idx % 10 == 0:
            print(round(W_loss.item(), 4), round(min_tl.item(), 4), round(opt_gap, 4), patience, pt_en)
            train_curve[j_idx][idx // 10] = W_loss.item()
            hard_train_curve[j_idx][idx // 10] = best_tl
            hashmap_size.append((len(hashmap), sum_thresh, min_tour))
            
        if opt_gap <= 0.1 or patience <= 0:
            print(f"Final: target_cost={cost:.4f}, best_tl={best_tl:.4f}, gap={opt_gap:.4f}")
            # Save last epoch best_perm as adjacency matrix if optimization ends early in eval mode
            if mode == "eval" and j_idx == 0 and 'last_epoch' not in eval_adjacency_matrices:
                eval_adjacency_matrices['last_epoch'] = convert_perm_to_adjacency(best_perm).detach().cpu().clone()
                print(f"Saved final epoch best_perm as adjacency matrix (early termination)")
            break
    
    print(f"=== Completed instance {j_idx+1}/{len(instances_to_process)} ===\n")

# Save adjacency matrices for eval mode
if mode == "eval":
    print("Saving best_perm adjacency matrices for eval mode...")
    adjacency_dir = "eval_adjacency_matrices"
    os.makedirs(adjacency_dir, exist_ok=True)
    
    # Save first epoch best_perm as adjacency matrix
    if 'first_epoch' in eval_adjacency_matrices:
        first_epoch_file = f"{adjacency_dir}/first_epoch_best_perm_{alg}_{S}_{lr}_{num_terms}_{device_idx}.pt"
        torch.save(eval_adjacency_matrices['first_epoch'], first_epoch_file)
        print(f"Saved first epoch best_perm adjacency matrix: {first_epoch_file}")
    
    # Save last epoch best_perm as adjacency matrix
    if 'last_epoch' in eval_adjacency_matrices:
        last_epoch_file = f"{adjacency_dir}/last_epoch_best_perm_{alg}_{S}_{lr}_{num_terms}_{device_idx}.pt"
        torch.save(eval_adjacency_matrices['last_epoch'], last_epoch_file)
        print(f"Saved last epoch best_perm adjacency matrix: {last_epoch_file}")
    
    print(f"Best_perm adjacency matrices saved in directory: {adjacency_dir}")

# Save results with adjacency suffix to distinguish from original
print("Saving results...")
train_curve_file = f"train_be_new/train_curve_{lr}_{num_terms}_{alg}_{S}_{mode}_{device_idx}_adjacency.npy"
hard_train_curve_file = f"train_be_new/hard_train_curve_{lr}_{num_terms}_{alg}_{S}_{mode}_{device_idx}_adjacency.npy"
hashmap_size_file = f"train_be_new/hashmap_size_{lr}_{num_terms}_{alg}_{S}_{mode}_{device_idx}_adjacency.pt"

os.makedirs("train_be_new", exist_ok=True)
np.save(train_curve_file, train_curve)
np.save(hard_train_curve_file, hard_train_curve)
torch.save(hashmap_size, hashmap_size_file)

print(f"Results saved:")
print(f"  - Train curve: {train_curve_file}")
print(f"  - Hard train curve: {hard_train_curve_file}")
print(f"  - Hashmap size: {hashmap_size_file}")
print(f"Optimization completed! Check log file: {log_filename}") 