import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from qap.IPF import *
from dimsum_objective import *
from dimsum import *
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
from collections import OrderedDict
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import maximum_bipartite_matching



S = str(sys.argv[2]) 
alg = str(sys.argv[1])
lr = float(sys.argv[3])
device_idx = int(sys.argv[4])
#device = torch.device(f"cuda:{device_idx}")
device = torch.device("cpu")
print(sys.argv)
T = 100
temp = 1000
target_num_terms = int(sys.argv[4])
num_terms = target_num_terms
target_num_input = 3
input_lst = []
data_lst = torch.load(f"input_data/data_lst_{target_num_terms}.pt")
cost_lst = np.load(f"input_data/cost_lst_{num_terms}_clock.npy")
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

idx = 0
while len(input_lst) < target_num_input:
    points, cost = data_lst[idx], 0#cost_lst[idx]
    num_terms = points.shape[0]
    input_lst.append((points, cost))
    idx += 1

class MatrixModel(nn.Module):
    def __init__(self, num_terms, alg):
        super().__init__()
        self.num_terms = num_terms
        self.mat = torch.nn.Linear(num_terms*2 -2 ,num_terms-1, bias=False)
        self.alg = alg
    def forward(self):
        W = torch.abs(self.mat.weight)
        for i in range(W.size()[0]):
            for j in range(W.size()[1]):
                if not self.num_terms + i > j:
                    W[i,j] = 0
        W = torch.softmax(W, dim=1)
        return W

assert len(train_curve) == target_num_input 
assert len(hashmap_size) == target_num_input
assert len(input_lst) == target_num_input
print(alg)

perms_lst = torch.load(f"input_data/greedy_{num_terms}_clock.pt")
trees_lst = [[] for idx in range(len(input_lst))]

for j_idx in tqdm(range(len(input_lst))):
    points, cost = input_lst[j_idx]
    cost = cost_lst[j_idx]
    
    #dimsum(greedytopo(points))

    perms = torch.abs(nn.Linear(num_terms*2 -2 ,num_terms-1).weight)
    for i in range(perms.size()[0]):
        for j in range(perms.size()[1]):
            if not num_terms + i > j:
                perms[i,j] = 0
    perms = ipf(perms, 5, 1,2)
    model = MatrixModel(num_terms, alg=alg).to(device) 
    hashmap = OrderedDict()
    state_dict = model.state_dict()
    state_dict['mat.weight'] = perms
    
    model.load_state_dict(state_dict)
    patience = 2000
    best_tl = num_terms**2
    if S == "greedy":
        perms = perms_lst[j_idx]
    best_perms = torch.clone(perms)
    best_node = None
    pt_en = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)    

    for idx in range(T):
        W = model.forward()
        t = temp / float(idx + 1)
        loss, perms, tl, sum_thresh, node = cont_Birkhoff_SFE(W, points, best_perms, hashmap, setting, alg, t)
        loss.backward()
         
        if tl < best_tl:
            best_tl = tl
            best_perms = torch.clone(perms)
            best_node = node
            pt_en = 0
            setting[1] = int(alg_lst[2])
            patience = 2000
            
            state_dict = model.state_dict()
            re_init = torch.abs(nn.Linear(num_terms*2 -2 ,num_terms-1).weight)
            for i in range(W.size()[0]):
                for j in range(W.size()[1]):
                    if not num_terms + i > j:
                        re_init[i,j] = 0
            re_init = ipf(re_init, 5, 1,2)
            state_dict['mat.weight'] = ipf(re_init, 5, 1,2)
            model.load_state_dict(state_dict)

        else:
            patience -= 1
            loss_gap = (abs(loss - tl).item())/(tl.item())
            if loss_gap <= 0.1:
                pt_en -= 0.01
            elif loss_gap >= 0.1:
                pt_en += 0.01

            if pt_en >= 1 and setting[1] <= 40:
                setting[1] = int(setting[1] * 1.3) + 1
                state_dict = model.state_dict()
                re_init = torch.abs(nn.Linear(num_terms*2 -2 ,num_terms-1).weight)
                for i in range(W.size()[0]):
                    for j in range(W.size()[1]):
                        if not num_terms + i > j:
                            re_init[i,j] = 0
                re_init = ipf(re_init, 5, 1,2)
                state_dict['mat.weight'] = ipf(re_init, 5, 1,2)
                model.load_state_dict(state_dict)
                pt_en = 0
 
            if pt_en <= -1 and setting[1] > 3:
                setting[1] -= 1
                pt_en = 0                

        opt_gap = abs(tl - cost).item()
        if idx // 10 > 0 and idx % 10 == 0:
            print(round(loss.item(), 4), tl, setting[1], round(opt_gap, 4), patience, pt_en)
            train_curve[j_idx][idx // 10] = loss.item()
            hard_train_curve[j_idx][idx // 10] = tl
            trees_lst[j_idx].append(node)
            #hashmap_size[j_idx][idx // 10] = (len(hashmap), setting[1], sum_thresh)
            
        if "pgd" in alg:
            for param in model.parameters():
                P = closest_tree(param.grad.data.cpu())
                weight = torch.sum(torch.abs(P*param.grad.data.cpu()))/num_terms**2
                if weight >= 1:
                    weight = 1
                param.data = (1 - lr*weight)*param.data + lr*weight*P
            
            torch.nn.Module.zero_grad(model)
            
        else:
            optimizer.step()
            optimizer.zero_grad()

        if patience <= 0:
            print(cost, best_tl)
            break

np.save(f"train_be_clock/train_curve_{lr}_{num_terms}_{alg}_{S}_{device_idx}.npy", train_curve)
np.save(f"train_be_clock/hard_train_curve_{lr}_{num_terms}_{alg}_{S}_{device_idx}.npy", hard_train_curve)
torch.save(trees_lst, f"train_be_clock/trees_records_{lr}_{num_terms}_{alg}_{S}_{device_idx}.pt")
#np.save(f"train_be_clock/hashmap_size_{lr}_{num_terms}_{alg}_{S}_{device_idx}.npy", hashmap_size)
