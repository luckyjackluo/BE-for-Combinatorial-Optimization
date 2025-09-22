import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from IPF import *
import matplotlib.pyplot as plt
import networkx as nx
import scipy

from pathlib import Path

import time
import logging
import os
import sys
import warnings
import itertools as it
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

from Birkhoff_TSP_old import *
from collections import OrderedDict
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import maximum_bipartite_matching
from lovasz import lovasz
from lovasz import edges_to_tourlength

S = str(sys.argv[2]) 
alg = str(sys.argv[1])
lr = float(sys.argv[3])
device_idx = int(sys.argv[-1])
print(device_idx)
#device = torch.device(f"cuda:{device_idx}")
device = torch.device("cpu")
print(sys.argv)
T = 10000
target_num_terms = int(sys.argv[4])
num_terms = target_num_terms
target_num_input = 20
gen = True
input_lst = []
data_lst = torch.load(f"../input_data/tsp/data_lst_{target_num_terms}.pt")
cost_lst = torch.load(f"../input_data/tsp/cost_lst_{target_num_terms}.pt")
dist_lst = torch.load(f"../input_data/tsp/dist_lst_{target_num_terms}.pt")
if "random" in S:
    data_lst = torch.load(f"../input_data/tsp/data_lst_{target_num_terms}_random.pt")
    cost_lst = torch.load(f"../input_data/tsp/cost_lst_{target_num_terms}_random.pt")
    dist_lst = torch.load(f"../input_data/tsp/dist_lst_{target_num_terms}_random.pt")

change_lst = []
train_curve = [[0 for idx in range(T//10)] for i in range(target_num_input)]
hard_train_curve = [[0 for idx in range(T//10)] for i in range(target_num_input)]
hashmap_size = [[(0, 0, 0) for idx in range(T//10)] for i in range(target_num_input)]
assert len(train_curve) == target_num_input 
assert len(hashmap_size) == target_num_input

alg_lst = alg.split("+")
if alg_lst[1] == "k":
    setting = [alg_lst[1], int(alg_lst[2])] 
elif alg_lst[1] == "p":
    setting = [alg_lst[1], float(alg_lst[2])]
print(alg_lst, setting)

if S == "predict":
    gen_perm_model.load_state_dict(torch.load("predict_s.model"))

idx = 0
if "qp" in S:
    qp_markov = np.load(f"train/final_sol_0.01_{num_terms}_pgd_random.npy")
    print("qp")
if S == "mst":
    mst_markov = torch.load(f"../input_data/tsp/W_{num_terms}_mst.pt")
if gen:
    while len(input_lst) < target_num_input:
        points, cost = data_lst[idx], cost_lst[idx]
        num_terms = points.shape[0]
        if "random" not in S:
            D = get_l2_dist(points) 
        else:
            D = dist_lst[idx]    
        input_lst.append((points, cost, D))
        idx += 1

    idx = 0
    torch.save(input_lst, f"../input_data/tsp/input_lst_{num_terms}.pt")
else:
    input_lst = torch.load(f"../input_data/tsp/input_lst_{num_terms}.pt")

# class MatrixModel(nn.Module):
#     def __init__(self, num_terms, alg):
#         super().__init__()
#         self.num_terms = num_terms
#         self.mat = torch.nn.Linear(num_terms, num_terms, bias=False)
#         self.alg = alg
#     def forward(self):
#         W = torch.nn.functional.softmax(torch.abs(self.mat.weight), dim=1)
#         return W

class MatrixModel(nn.Module):
    def __init__(self, num_terms, alg):
        super().__init__()
        self.num_terms = num_terms
        self.mat = torch.nn.Linear(num_terms, num_terms, bias=False)
        self.alg = alg
    def forward(self):
        if self.alg == "gd":
            W = torch.abs(self.mat.weight) 
            W = ipf(W, 10, 1,1)
        else:
            W = self.mat.weight
        return W
    
alg_lst = alg.split("+")
if alg_lst[1] == "k":
    setting = [alg_lst[1], int(alg_lst[2])] 
elif alg_lst[1] == "p":
    setting = [alg_lst[1], float(alg_lst[2])]
print(alg_lst, setting)

assert len(train_curve) == target_num_input 
assert len(hashmap_size) == target_num_input
assert len(input_lst) == target_num_input
print(alg)

for j_idx in tqdm.tqdm(range(len(input_lst))):
    points, cost, D = input_lst[j_idx]
    
    if "qp" in S:
        perms = torch.tensor(qp_markov[j_idx][0])
    elif "mst" in S:
        perms = mst_markov[j_idx]
    elif "constant" in S:
        perms = ipf(torch.ones(num_terms, num_terms)/num_terms, 5, 1, 1)
    
    model = MatrixModel(num_terms, alg=alg).to(device) 
    state_dict = model.state_dict()
    weight = torch.rand(num_terms, num_terms, device=device)
    weight = torch.abs(weight) / num_terms
    state_dict['mat.weight'] = ipf(weight, 5, 1, 1)
    
    model.load_state_dict(state_dict)
    hashmap = OrderedDict()
    patience = 2000
    best_tl = num_terms**2
    best_perms = perms.clone()
    pt_en = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)    
    
    # Initialize previous permutations set
    prev_all_perms = None
    # Initialize set to track unique permutations
    unique_perms = set()
    total_unique = 0

    def perm_to_hashable(perm):
        # Convert to numpy and then to a tuple of tuples for hashing
        return tuple(map(tuple, perm.detach().cpu().numpy()))

    for idx in range(T):
        W = model.forward() 
        tl, loss, perms, num_P, sum_thresh, all_perms = cont_Birkhoff_SFE(W, num_terms, D, best_perms, setting, alg, device=device)
        loss = loss.to(device)
        
        # Compare with previous permutations if they exist
        if prev_all_perms is not None:
            # Count different permutations
            diff_count = 0
            for i, (curr_perm, prev_perm) in enumerate(zip(all_perms, prev_all_perms)):
                if not torch.allclose(curr_perm, prev_perm):
                    diff_count += 1
            
            # Count new unique permutations
            new_unique = 0
            for perm in all_perms:
                # Convert permutation matrix to hashable format using numpy
                perm_hash = perm_to_hashable(perm)
                if perm_hash not in unique_perms:
                    unique_perms.add(perm_hash)
                    new_unique += 1
                    total_unique += 1
            
            if idx % 10 == 0:  # Print every 10 iterations to avoid too much output
                # Calculate tour lengths for all permutations
                tour_lengths = [objective_function(perm, D, num_terms).item() for perm in all_perms]
                print(f"iter {idx}: {diff_count}/{len(all_perms)} perms changed, {new_unique} new unique, total unique: {total_unique}")
                print(f"Tour lengths: {[round(tl, 4) for tl in tour_lengths]}")
            
            # Store current permutations for next comparison
        prev_all_perms = [p.clone() for p in all_perms]
        
        loss.backward()
         
        if tl < best_tl:
            best_tl = tl
            #best_perms = perms
            pt_en = 0
            setting[1] = int(alg_lst[2])
            patience = 2000
            
            #state_dict = model.state_dict()
            #weight = torch.rand(num_terms, num_terms, device=device)
            #weight = torch.abs(weight) / num_terms
            #state_dict['mat.weight'] = ipf(weight, 5, 1, 1)
            #model.load_state_dict(state_dict)

        else:
            patience -= 1
            loss_gap = (abs(loss - tl).item())/(tl.item())
            if loss_gap <= 0.05:
                pt_en += 0.01
            elif loss_gap <= 0.05:
                pt_en -= 0.01

            if pt_en >= 1 and setting[1] <= 20:
                setting[1] = int(setting[1] * 1.3) + 1
                # state_dict = model.state_dict()
                # weight = torch.rand(num_terms, num_terms, device=device)
                # weight = torch.abs(weight) / num_terms
                # state_dict['mat.weight'] = ipf(weight, 5, 1, 1)
                # model.load_state_dict(state_dict)
                # pt_en = 0
 
            if pt_en <= -1 and setting[1] > 1:
                setting[1] -= 1
                pt_en = 0                

        opt_gap = abs(tl.item() - cost)
        if idx // 10 > 0 and idx % 10 == 0:
            print(round(loss.item(), 4), round(tl.item(), 4), round(best_tl.item(), 4), num_P, round(opt_gap, 4), patience, pt_en)
            print("==============================================================================================================")
            train_curve[j_idx][idx // 10] = loss.item()
            hard_train_curve[j_idx][idx // 10] = tl
            hashmap_size[j_idx][idx // 10] = (len(hashmap), num_P, sum_thresh)
            
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
                for i, j in zip(row, col):
                    P[i, j] = 1
                
                # Adaptive step size
                grad_norm = torch.norm(grad)
                if grad_norm > 0:
                    step_size = lr / (1 + grad_norm)
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

        if opt_gap <= 0.001 or patience <= 0:
            print(cost, best_tl)
            break

np.save(f"train_be/train_curve_{lr}_{num_terms}_{alg}_{S}_{device_idx}.npy", train_curve)
np.save(f"train_be/hard_train_curve_{lr}_{num_terms}_{alg}_{S}_{device_idx}.npy", hard_train_curve)
np.save(f"train_be/hashmap_size_{lr}_{num_terms}_{alg}_{S}_{device_idx}.npy", hashmap_size)
