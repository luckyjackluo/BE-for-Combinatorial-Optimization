import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from qap.IPF import *
import matplotlib.pyplot as plt
import networkx as nx
import networkx
from itertools import combinations
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
torch.set_printoptions(sci_mode = False , precision = 3)
np.set_printoptions(precision=3, suppress = True)
sys.path.append("../set2graph")

from set2graph import PointsetToGraphModel

from tqdm import tqdm

from Birkhoff_TSP_old import *
import sys
import logging
import math
import random
from collections import OrderedDict
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import maximum_bipartite_matching
from torch.utils.data import Dataset

S = str(sys.argv[2]) 
alg = str(sys.argv[1])
lr = float(sys.argv[3])
device_idx = int(sys.argv[-1])
device = torch.device(f"cuda:{device_idx}")
print(sys.argv)    
target_num_terms = int(sys.argv[4])
target_num_input = 20
num_terms = target_num_terms
T = 10000
gen = True

print(S, alg, lr, device_idx, device)

alg_lst = alg.split("+")
if alg_lst[1] == "k":
    setting = [alg_lst[1], int(alg_lst[2])] # Type and Cap
elif alg_lst[1] == "p":
    setting = [alg_lst[1], float(alg_lst[2])]
print(alg_lst, setting)

data_lst = torch.load(f"../input_data/tsp/data_lst_{target_num_terms}.pt")
cost_lst = torch.load(f"../input_data/tsp/cost_lst_{target_num_terms}.pt")
dist_lst = torch.load(f"../input_data/tsp/dist_lst_{target_num_terms}.pt")
if S == "random":
    data_lst = torch.load(f"../input_data/tsp/data_lst_{target_num_terms}_random.pt")
    cost_lst = torch.load(f"../input_data/tsp/cost_lst_{target_num_terms}_random.pt")
    dist_lst = torch.load(f"../input_data/tsp/dist_lst_{target_num_terms}_random.pt")
    

def objective_function(P, D, n):
    obj = 0
    for i in range(n-1):
        obj += torch.matmul(torch.matmul(P[:, i], D), P[:, i+1])
    obj += torch.matmul(torch.matmul(P[:, i+1], D), P[:, 0])
    return obj

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

input_lst = []
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

from set2graph.models import NewPointsetToGraphModel

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class MatrixModel(nn.Module):
    def __init__(self, num_terms):
        super().__init__()
        self.num_terms = num_terms
        self.mat = NewPointsetToGraphModel(
             symmetric=False,
             num_features=2,
             hidden_channels=64,
             one_to_one_config={"type": "Transformer"},
             one_to_two_config={"type": "Universal_1to2"}
             )
        self.decoder = SimpleMLP(64, 64, 1)
    def forward(self, points):
        output = torch.abs(self.decoder(self.mat(points)).data)
        W = output.reshape(self.num_terms, self.num_terms)
        #perms = output[:, 1].reshape(self.num_terms, self.num_terms)
        W = ipf(W, self.num_terms, 1,1)
        #perms = ipf(perms, self.num_terms, 1,1)
        return W#, perms

#class MatrixModel(nn.Module):
#    def __init__(self, num_terms):
#        super().__init__()
#        self.num_terms = num_terms
#        self.mat = NewPointsetToGraphModel(
#             symmetric=False,
#             num_features=2,
#             hidden_channels=64,
#             one_to_one_config={"type": "Transformer"},
#             one_to_two_config={"type": "Universal_1to2"}
#             )
#        self.score_encoder = SimpleMLP(1, 64, 64)
#        self.decoder = SimpleMLP(64 + 64, 256, 1)
#    def forward(self, points, score):
#        points_output = self.mat(points).data
#        score_output = self.score_encoder(score)
#        W = torch.abs(self.decoder(torch.concat([points_output, score_output], dim=1)))
#        W = W.reshape(self.num_terms, self.num_terms)
#        W = ipf(W, self.num_terms, 1,1)
#        return W


class TSPDataset(Dataset):
    def __init__(self, data_lst, dist_lst, best_perms_lst, cost_lst):
        self.data_lst = data_lst
        self.dist_lst = dist_lst
        self.best_perms_lst = best_perms_lst
        self.cost_lst = cost_lst

    def __len__(self):
        return len(self.data_lst)

    def __getitem__(self, idx):
        return self.data_lst[idx], self.dist_lst[idx], self.best_perms_lst[idx], cost_lst[idx]

    def update(self, idx, best_perms):
        self.best_perms_lst[idx] = best_perms
        
model = MatrixModel(num_terms).to(device)
#model.load_state_dict(torch.load(f"train_nn/best_nn_{0.001}_{num_terms}_{alg}_constant_{100}.model"))
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
hashmap = OrderedDict()
best_perms_lst = []
best_tl_lst = [num_terms**2 for idx in range(target_num_input)]
data_lst = []
dist_lst = []
cost_lst = []

for j_idx in tqdm.tqdm(range(len(input_lst))):
    points, cost, D = input_lst[j_idx]
    if "qp" in S:
        perms = torch.tensor(qp_markov[j_idx][0])
    elif "mst" in S:
        perms = mst_markov[j_idx]
    elif "constant" in S:
        perms = ipf(torch.ones(num_terms, num_terms)/num_terms, 5, 1, 1)

    data_lst.append(points)
    dist_lst.append(D)
    best_perms_lst.append(perms)
    cost_lst.append(cost)

new_dataset = TSPDataset(data_lst, dist_lst, best_perms_lst, cost_lst)
mean_train_curve = []
mean_hard_train_curve = []

for idx in range(T):
    total_loss = 0
    for j_idx in tqdm.tqdm(range(len(input_lst))):
        #for inner_idx in range(1):
        points, D, best_perms, cost = new_dataset[j_idx]  
            #W = model(points.to(device), best_perms.reshape(num_terms**2, 1).to(device))
        W = model(points.to(device))
        tl, loss, perms, num_P, sum_thresh =  cont_Birkhoff_SFE(W.cpu(), num_terms**2, D, best_perms, hashmap, setting, alg, device=device) 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if tl < best_tl_lst[j_idx]:
            best_tl_lst[j_idx] = tl
            new_dataset.update(j_idx, perms)
            #print(tl, inner_idx)

    total_loss += loss.item()
    # total_loss.backward()
    # optimizer.step()
    # optimizer.zero_grad()
    
    print(np.mean(best_tl_lst), idx)
    mean_train_curve.append(total_loss/len(input_lst))
    mean_hard_train_curve.append(np.mean(best_tl_lst))

np.save(f"train_nn/mean_train_curve_{lr}_{num_terms}_{alg}_{S}_{device_idx}.npy", mean_train_curve)
np.save(f"train_nn/mean_hard_train_curve_{lr}_{num_terms}_{alg}_{S}_{device_idx}.npy", mean_hard_train_curve)
