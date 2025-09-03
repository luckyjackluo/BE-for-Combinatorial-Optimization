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
target_num_input = 100
num_terms = target_num_terms

print(S, alg, lr, device_idx, device)

alg_lst = alg.split("+")
if alg_lst[1] == "k":
    setting = [alg_lst[1], int(alg_lst[2])] # Type and Cap
elif alg_lst[1] == "p":
    setting = [alg_lst[1], float(alg_lst[2])]
print(alg_lst, setting)

# data_lst = torch.load(f"input_data/data_lst_{target_num_terms}.pt")
# cost_lst = torch.load(f"input_data/cost_lst_{target_num_terms}.pt")
# dist_lst = torch.load(f"input_data/dist_lst_{target_num_terms}.pt")
# if S == "random":
#     data_lst = torch.load(f"input_data/data_lst_{target_num_terms}_random.pt")
#     cost_lst = torch.load(f"input_data/cost_lst_{target_num_terms}_random.pt")
#     dist_lst = torch.load(f"input_data/dist_lst_{target_num_terms}_random.pt")
    

def objective_function(P, D, n):
    obj = 0
    for i in range(n-1):
        obj += torch.matmul(torch.matmul(P[:, i], D), P[:, i+1])
    obj += torch.matmul(torch.matmul(P[:, i+1], D), P[:, 0])
    return obj


class MatrixModel(nn.Module):
    def __init__(self, num_terms, alg="gd"):
        super().__init__()
        self.num_terms = num_terms
        self.mat = nn.Linear(self.num_terms,self.num_terms, bias=False)
        self.alg = alg
    def forward(self):
        if self.alg == "gd":
            W = torch.abs(self.mat.weight)
            W = ipf(W, self.num_terms, 1,1)
        else:
            W = self.mat.weight
        return W

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
             num_features=2+num_terms,
             hidden_channels=128,
             one_to_one_config={"type": "Transformer"},
             one_to_two_config={"type": "Universal_1to2"}
             )
        self.decoder = SimpleMLP(128, 256, 1)
    def forward(self, points):
        output = torch.abs(self.decoder(self.mat(points)).data)
        W = output.reshape(self.num_terms, self.num_terms)
        #perms = output[:, 1].reshape(self.num_terms, self.num_terms)
        W = ipf(W, self.num_terms, 1,1)
        #perms = ipf(perms, self.num_terms, 1,1)
        return W#, perms

class TSPDataset(Dataset):
    def __init__(self, data_lst, dist_lst, best_perms_lst):
        self.data_lst = data_lst
        self.dist_lst = dist_lst
        self.best_perms_lst = best_perms_lst

    def __len__(self):
        return len(self.data_lst)

    def __getitem__(self, idx):
        return self.data_lst[idx], self.dist_lst[idx], self.best_perms_lst[idx]

    def update(self, idx, best_perms):
        self.best_perms_lst[idx] = best_perms
        
model = MatrixModel(num_terms).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

while True:
    hashmap = OrderedDict()
    best_perms_lst = []
    best_tl_lst = [num_terms**2 for idx in range(target_num_input)]
    data_lst = []
    dist_lst = []
    npoints = num_terms
    for epoch in tqdm.tqdm(range(target_num_input)):
        nodes = list(range(npoints))
        data = generate_points(npoints)
        points = data.numpy()
        distances = {
            (i, j): math.sqrt(sum((points[i][k] - points[j][k]) ** 2 for k in range(2)))
            for i, j in combinations(nodes, 2)
        }
    
        dist = torch.zeros(npoints, npoints)
        for key, items in distances.items():
            dist[key[0], key[1]] = items
            dist[key[1], key[0]] = items
    
        points = data
        distances = dist
        g = nx.complete_graph(num_terms)
        for i in range(distances.shape[0]):
            for j in range(i, distances.shape[1]):
                if i == j:
                    continue
                else:
                    g[i][j]['weight'] = distances[i][j]
        mst = nx.minimum_spanning_tree(g)
        tour = list(nx.dfs_preorder_nodes(mst))
        P = torch.zeros(num_terms, num_terms)
        for tour_idx in range(len(tour)):
            P[tour[tour_idx], tour_idx] = 1
        init_perm = torch.zeros((num_terms, num_terms))
        for pos in range(len(tour)):
            city_idx = tour[pos]
            init_perm[city_idx][pos] = 1
        
        data_lst.append(points)
        dist_lst.append(distances)
        best_perms_lst.append(init_perm)

    new_dataset = TSPDataset(data_lst, dist_lst, best_perms_lst)
    outer_patience = [10 for i in range(target_num_input)]

    while True:
        total_loss = 0
        #outer_patience = [5 for i in range(target_num_input)]
        
        for j_idx in tqdm.tqdm(range(len(new_dataset))):
            points, D, best_perms = new_dataset[j_idx] 
            #points = points.to(device)
            #D = D.to(device)
            #best_perms = best_perms.to(device)
            #inner_patience = 10

            W = model(torch.concat([points, best_perms], dim=1).to(device))
            tl, loss, perms, num_P, sum_thresh =  cont_Birkhoff_SFE(W.cpu(), num_terms**2 - num_terms, D, best_perms, hashmap, setting, alg, device=device) 
            total_loss += loss

            if tl < best_tl_lst[j_idx]:
                outer_patience[j_idx] += 10 #[5 for i in range(target_num_input)]
                best_tl_lst[j_idx] = tl
                new_dataset.update(j_idx, perms)

            else:
                #inner_patience -= 1
                if outer_patience[j_idx] > 0:
                    outer_patience[j_idx] -= 1

            #print(inner_patience)
            #if inner_patience <= 0:
                #outer_patience -= 1
                #break
        
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(np.mean(best_tl_lst), sum(outer_patience))
        torch.save(model.state_dict(), f"train_nn/best_nn_{lr}_{num_terms}_{alg}_{S}_{target_num_input}.model")

        if sum(outer_patience) <= 0:
            break
