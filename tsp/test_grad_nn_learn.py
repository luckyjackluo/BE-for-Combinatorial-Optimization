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
num_epoch = int(sys.argv[4])
target_num_input = 2100
#num_terms = target_num_terms

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

import wandb
wandb.init(
            # set the wandb project where this run will be logged
    project="NN-TSP",

                    # track hyperparameters and run metadata
    config={
    "learning_rate": lr, 
    "alg": alg,
    "S": S,
    "device_idx": device_idx,
    }
)

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

class NNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mat = NewPointsetToGraphModel(
             symmetric=False,
             num_features=2,
             hidden_channels=128,
             one_to_one_config={"type": "Transformer"},
             one_to_two_config={"type": "Linear_1to2"}
             )
        #self.score_encoder = SimpleMLP(1, 32, 32)
        self.decoder = SimpleMLP(128, 128, 1)
    def forward(self, points, score, num_terms):
        points_output = self.mat(points).data
        #score_output = self.score_encoder(score)
        #W = torch.abs(self.decoder(torch.concat([points_output, score_output], dim=-1)))
        W = torch.abs(self.decoder(points_output))
        W = W.reshape(num_terms, num_terms)
        W = torch.nn.functional.softmax(W, dim=-1)
        #W = ipf(W, 10, 1, 1)
        return W

class TSPDataset(Dataset):
    def __init__(self, data_lst, dist_lst, best_perms_lst, mst_tl_lst, best_tl_lst):
        self.data_lst = data_lst
        self.dist_lst = dist_lst
        self.best_perms_lst = best_perms_lst
        self.mst_tl_lst = mst_tl_lst
        self.best_tl_lst = best_tl_lst

    def __len__(self):
        return len(self.data_lst)

    def __getitem__(self, idx):
        return self.data_lst[idx], self.dist_lst[idx], self.best_perms_lst[idx], self.mst_tl_lst[idx]

    def update(self, idx, best_perms, best_tl):
        self.best_perms_lst[idx] = best_perms
        self.best_tl_lst[idx] = best_tl
        
model = NNModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#while True:
dataset_lst = []
for npoints in [8, 10, 12, 15, 20]:
    if npoints == 20:
        target_num_input = 100
    hashmap = OrderedDict()
    best_perms_lst = []
    best_tl_lst = [npoints**2 for idx in range(target_num_input)]
    data_lst = []
    dist_lst = []
    mst_tl_lst = []
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
        g = nx.complete_graph(npoints)
        for i in range(distances.shape[0]):
            for j in range(i, distances.shape[1]):
                if i == j:
                    continue
                else:
                    g[i][j]['weight'] = distances[i][j]
        mst = nx.minimum_spanning_tree(g)
        tour = list(nx.dfs_preorder_nodes(mst))
        P = torch.zeros(npoints, npoints)
        for tour_idx in range(len(tour)):
            P[tour[tour_idx], tour_idx] = 1

        tl_mst = objective_function(P, dist, npoints)

        init_perm = torch.zeros((npoints, npoints))
        for pos in range(len(tour)):
            city_idx = tour[pos]
            init_perm[city_idx][pos] = 1
        
        data_lst.append(points)
        dist_lst.append(distances)
        mst_tl_lst.append(tl_mst)
        best_perms_lst.append(torch.ones((npoints, npoints)))

    dataset_lst.append(TSPDataset(data_lst, dist_lst, best_perms_lst, mst_tl_lst, best_tl_lst))

#while True:
for t in range(num_epoch):
    for batch in range(10):
        total_loss = 0
        total_tl = 0
        total_tl_mst = 0
        dataset_index_lst = torch.randperm(4)
        for dataset_index in dataset_index_lst:
            new_dataset = dataset_lst[dataset_index]
            for j_idx in tqdm.tqdm(range(batch*100, (batch+1)*100)):
            #for dataset_index in dataset_index_lst:
                new_dataset = dataset_lst[dataset_index]
                points, D, best_perms, mst_tl = new_dataset[j_idx] 
                num_terms = len(points)
                input_perms = best_perms.flatten().unsqueeze(1)
                W = model(points.to(device), input_perms.to(device), num_terms)
                ds_loss = sum((torch.sum(W, dim=0) - 1)**2)
                tl, loss, perms, num_P, sum_thresh =  cont_Birkhoff_SFE(W.cpu(), num_terms, D, best_perms, setting, device=device) 
                loss = loss.to(device)
                loss += ds_loss

                if tl < new_dataset.best_tl_lst[j_idx]:
                    #best_tl_lst[j_idx] = tl
                    new_dataset.update(j_idx, perms, tl)
                    
                total_loss += loss
                total_tl += tl
                total_tl_mst += mst_tl
        
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(total_loss, total_tl)
        
    for batch in range(20, 21):
        total_loss = 0
        total_tl = 0
        total_tl_mst = 0
        num_t = 0
        dataset_index_lst = torch.randperm(4)
        for dataset_index in dataset_index_lst:
            new_dataset = dataset_lst[dataset_index]
            for j_idx in tqdm.tqdm(range(batch*100, (batch+1)*100)): 
                points, D, best_perms, mst_tl = new_dataset[j_idx] 
                num_terms = len(points)
                input_perms = best_perms.flatten().unsqueeze(1)
                W = model(points.to(device), input_perms.to(device), num_terms)
                #ds_loss = sum((torch.sum(W, dim=0) - 1)**2)
                tl, loss, perms, num_P, sum_thresh =  cont_Birkhoff_SFE(W.cpu(), num_terms, D, best_perms, setting, device=device) 
                #loss = loss.to(device)
                #loss += 10*ds_loss
                
                if tl < new_dataset.best_tl_lst[j_idx]:
                    #best_tl_lst[j_idx] = tl
                    new_dataset.update(j_idx, perms, tl)

                total_loss += loss
                total_tl += tl
                total_tl_mst += mst_tl
                num_t += 1

        loss = total_loss/num_t
        tl = total_tl/num_t
        tl_mst = total_tl_mst/num_t

    wandb.log({"loss": loss.item(), "tl": tl.item(), "gap": (tl.item() - tl_mst.item())/tl_mst.item()})
    torch.save(model.state_dict(), f"train_nn/best_nn_{lr}_{num_epoch}_{alg}_{S}_{target_num_input}.model")
    
    
    for batch in range(1):
       total_loss = 0
       total_tl = 0
       total_tl_mst = 0
       num_t = 0
       new_dataset = dataset_lst[-1]
       for j_idx in tqdm.tqdm(range(batch*100, (batch+1)*100)):
           points, D, best_perms, mst_tl = new_dataset[j_idx] 
           num_terms = len(points)
           input_perms = best_perms.flatten().unsqueeze(1)
           W = model(points.to(device), input_perms.to(device), num_terms)
           #ds_loss = sum((torch.sum(W, dim=0) - 1)**2)
           tl, loss, perms, num_P, sum_thresh =  cont_Birkhoff_SFE(W.cpu(), num_terms, D, best_perms, setting, device=device) 
           #loss = loss.to(device)
           #loss += 10*ds_loss
           
           if tl < best_tl_lst[j_idx]:
               #best_tl_lst[j_idx] = tl
               new_dataset.update(j_idx, perms, tl)
    
           total_loss += loss
           total_tl += tl
           total_tl_mst += mst_tl
           num_t += 1
    
       loss = total_loss/num_t
       tl = total_tl/num_t
       tl_mst = total_tl_mst/num_t
    
    wandb.log({"gen_loss": loss.item(), "gen_tl": tl.item(), "gen_gap": (tl.item() - tl_mst.item())/tl_mst.item()})
    
    
