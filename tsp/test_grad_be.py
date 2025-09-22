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

from Birkhoff_TSP import *
from collections import OrderedDict
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import maximum_bipartite_matching

S = str(sys.argv[2]) 
alg = str(sys.argv[1])
lr = float(sys.argv[3])
device_idx = int(sys.argv[-1])
print(device_idx)
#device = torch.device(f"cuda:{device_idx}")
device = torch.device("cpu")
print(sys.argv)
T = 2000
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
    mst_cost_lst = torch.load(f"../input_data/tsp/cost_lst_{target_num_terms}_mst_random.pt")
    dist_lst = torch.load(f"../input_data/tsp/dist_lst_{target_num_terms}_random.pt")

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

if S == "predict":
    gen_perm_model.load_state_dict(torch.load("predict_s.model"))

idx = 0
if "qp" in S:
    qp_markov = np.load(f"train/final_sol_0.01_{num_terms}_pgd_random.npy")
    print("qp")
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

class MatrixModel(nn.Module):
    def __init__(self, num_terms):
        super().__init__()
        self.num_terms = num_terms
        self.mat = torch.nn.Linear(num_terms, num_terms, bias=False)
    def forward(self):
        W = torch.abs(self.mat.weight) 
        W = ipf(W, 10, 1,1)
        return W

alg_lst = alg.split("+")
if alg_lst[1] == "k":
    setting = [alg_lst[1], int(alg_lst[2])] 
elif alg_lst[1] == "p":
    setting = [alg_lst[1], float(alg_lst[2])]
print(alg_lst, setting)

assert len(train_curve) == target_num_input 
assert len(input_lst) == target_num_input
print(alg)

for j_idx in tqdm.tqdm(range(len(input_lst))):
    points, cost, D = input_lst[j_idx]    
    # if "qp" in S:
    #     perms = torch.tensor(qp_markov[j_idx][0])
    # elif "mst" in S:
    #     perms = mst_markov[j_idx]
    # elif "constant" in S:
    #     perms = ipf(torch.ones(num_terms, num_terms)/num_terms, 5, 1, 1)
    
    model_W = MatrixModel(num_terms).to(device) 
    state_dict = model_W.state_dict()
    state_dict['mat.weight'] = ipf(torch.ones(num_terms, num_terms)/num_terms, 5, 1, 1)
    model_W.load_state_dict(state_dict)
    optimizer_W = torch.optim.Adam(model_W.parameters(), lr=0.01)

    model_perms = MatrixModel(num_terms).to(device) 
    state_dict = model_perms.state_dict()
    state_dict['mat.weight'] = ipf(torch.ones(num_terms, num_terms)/num_terms, 5, 1, 1)
    model_perms.load_state_dict(state_dict)
    optimizer_perms = torch.optim.Adam(model_perms.parameters(), lr=0.01)
    
    hashmap = OrderedDict()
    patience = 4000
    best_tl = num_terms**2
    pt_en = 0 

    for idx in range(T):
        perms = model_perms()
        total_perms_loss = 0

        for inner_idx in range(10):
            W = model_W()
            W_loss, perms_loss, min_tl, sum_thresh, min_tour = cont_Birkhoff_SFE(W, 5, D, perms, hashmap, setting, alg, device=device)
            W_loss.backward()
            total_perms_loss += perms_loss
            optimizer_W.step()
            optimizer_W.zero_grad()
         
            if min_tl < best_tl:
                best_tl = min_tl
                pt_en = 0
                # state_dict = model_W.state_dict()
                # state_dict['mat.weight'] = ipf(torch.ones(num_terms, num_terms)/num_terms, 5, 1, 1)
                # model_W.load_state_dict(state_dict)
                # patience = 4000
            else:
                patience -= 1
                #loss_gap = (abs(W_loss - min_tl).item())/(min_tl.item())
                #if loss_gap <= 0.05:
                #    pt_en += 0.01
                #elif loss_gap <= 0.05:
                #    pt_en -= 0.01

            #if pt_en >= 1 and setting[1] <= 20:
                #setting[1] = int(setting[1] * 1.3) + 1
                #pt_en = 0
 
            #if pt_en <= -1 and setting[1] > 1:
                #setting[1] -= 1
                #pt_en = 0    

        total_perms_loss += objective_function(perms, D, num_terms)**2
        total_perms_loss.backward()
        optimizer_perms.step()
        optimizer_perms.zero_grad()

        opt_gap = abs(min_tl.item() - cost)
        
        if idx // 10 > 0 and idx % 10 == 0:
            print(round(W_loss.item(), 4), round(min_tl.item(), 4), round(opt_gap, 4), patience, pt_en)
            train_curve[j_idx][idx // 10] = W_loss.item()
            hard_train_curve[j_idx][idx // 10] = best_tl
            hashmap_size.append((len(hashmap), sum_thresh, min_tour))
            
        if opt_gap <= 0.1 or patience <= 0:
            print(cost, best_tl)
            break

np.save(f"train_be_new/train_curve_{lr}_{num_terms}_{alg}_{S}_{device_idx}.npy", train_curve)
np.save(f"train_be_new/hard_train_curve_{lr}_{num_terms}_{alg}_{S}_{device_idx}.npy", hard_train_curve)
torch.save(hashmap_size, f"train_be_new/hashmap_size_{lr}_{num_terms}_{alg}_{S}_{device_idx}.pt")
