import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from qap.IPF import *
#from dimsum import *
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
from ml_lib.feature_specification import FeatureSpecification, MSEFeature
from ml_lib.datasets.base_classes import Dataset
from ml_lib.pipeline.trainer import Trainer
from ml_lib.pipeline.training_hooks import TqdmHook

torch.set_printoptions(sci_mode = False , precision = 3)
np.set_printoptions(precision=3, suppress = True)
sys.path.append("../set2graph")

from set2graph import PointsetToGraphModel

from experiments.dataset import MSTDataset

from tqdm import tqdm

from Birkhoff_TSP_old import *

import sys
import logging
import math
import random
from collections import OrderedDict
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import maximum_bipartite_matching

S = str(sys.argv[2]) 
alg = str(sys.argv[1])
lr = float(sys.argv[3])
#device_idx = int(sys.argv[4])
#device = torch.device(f"cuda:{device_idx}")
device = torch.device("cpu")
print(sys.argv)
T = 10000
target_num_terms = int(sys.argv[4])
num_terms = target_num_terms
target_num_input = 20
gen = True
input_lst = []
if "random" in S:
    data_lst = torch.load(f"data_lst_{target_num_terms}_random.pt")
    perm_lst = torch.load(f"perm_lst_{target_num_terms}_random.pt")
    cost_lst = torch.load(f"cost_lst_{target_num_terms}_random.pt")
    dist_lst = torch.load(f"dist_lst_{target_num_terms}_random.pt")
else:
    data_lst = torch.load(f"data_lst_{target_num_terms}.pt")
    perm_lst = torch.load(f"perm_lst_{target_num_terms}.pt")
    cost_lst = torch.load(f"cost_lst_{target_num_terms}.pt")
    #dist_lst = torch.load(f"dist_lst_{target_num_terms}_random.pt")

change_lst = []
train_curve = [[0 for idx in range(T//10)] for i in range(target_num_input)]
hard_train_curve = [[0 for idx in range(T//10)] for i in range(target_num_input)]
hashmap_size = [[(0, 0, 0) for idx in range(T//10)] for i in range(target_num_input)]
assert len(train_curve) == target_num_input 
assert len(hashmap_size) == target_num_input

alg_lst = alg.split("+")
if alg_lst[1] == "k":
    setting = [alg_lst[1], int(alg_lst[2])] # Type and Cap
elif alg_lst[1] == "p":
    setting = [alg_lst[1], float(alg_lst[2])]
print(alg_lst, setting)

if S == "predict":
    gen_perm_model.load_state_dict(torch.load("predict_s.model"))

idx = 0
if "qp" in S:
    try:
        qp_markov = np.load(f"train/final_sol_0.01_{num_terms}_pgd_markov.npy")
    except:
        qp_markov = np.load(f"train/final_sol_0.01_{num_terms}_pgd_random.npy")
    
    if "random" in S:
        print("random+qp")
        qp_markov = np.load(f"train_random/final_sol_0.01_{num_terms}_pgd_random.npy")

if S == "mst":
    mst_markov = torch.load(f"W_{num_terms}_mst.pt")
if gen:
    while len(input_lst) < target_num_input:
        points, perms = data_lst[idx], perm_lst[idx]
        cost = cost_lst[idx]
        perms = ipf(perms, 100)
        num_terms = points.shape[0]
        if "random" in S:
            D = dist_lst[idx]
        else:
            D = get_l2_dist(points)
        #W_init_lst.append(perms)

        #if S == 'rand':                          
        #    perms = torch.rand((num_terms, num_terms))

        #elif S == 'l1':
        #    perms = get_l1_dist(points)

        #elif S == 'l2':
        #    perms = D

        #elif S == 'constant':
        #    perms = torch.ones_like(perms)/num_terms

        #elif S == 'predict':
        #    n = num_terms
        #    perms = gen_perm_model(points)
        #    perms = perms.detach()
        #    result_lst.append(objective_function(perms.numpy(), D, n))
        input_lst.append((points, perms, D, cost))
        idx += 1
    idx = 0

else:
    input_lst = torch.load("input_lst.pt")

class MatrixModel(nn.Module):
    def __init__(self, num_terms, alg="gd"):
        super().__init__()
        self.num_terms = num_terms
        self.mat = nn.Linear(self.num_terms,self.num_terms, bias=False) 
        self.alg = alg
    def forward(self):
        if self.alg == "gd":
            W = torch.abs(self.mat.weight)
            W = ipf(W, self.num_terms*2, 1,1)
        else:
            W = self.mat.weight
        return W

alg_lst = alg.split("+")
if alg_lst[1] == "k":
    setting = [alg_lst[1], int(alg_lst[2])] # Type and Cap
elif alg_lst[1] == "p":
    setting = [alg_lst[1], float(alg_lst[2])]
print(alg_lst, setting)

assert len(train_curve) == target_num_input 
assert len(hashmap_size) == target_num_input
assert len(input_lst) == target_num_input
print(alg)

for j_idx in tqdm.tqdm(range(len(input_lst))):
    points, perms, D, cost = input_lst[j_idx] 
    model = MatrixModel(num_terms, alg=alg).to(device) 
    state_dict = model.state_dict()
    state_dict['mat.weight'] = perms 
    model.load_state_dict(state_dict)
    if "qp" in S:
        "qp"
        perms = torch.tensor(qp_markov[j_idx][0])
    elif S == "mst":
        perms = mst_markov[j_idx]
    elif S == "constant":
        perms = torch.ones_like(perms)/num_terms
    
    hashmap = OrderedDict()
    patience = 10
    best_tl = num_terms**2
    for idx in range(T):
        W = model.forward()
        tl, loss, perms, num_P, sum_thresh =  cont_Birkhoff_SFE(W.cpu(), num_terms**2 - num_terms, D, perms, hashmap, setting, alg)
        loss.backward()
        
        if tl < best_tl:
            best_tl = tl
            pt_en = 0
            patience = 2000    
            
        
        opt_gap = abs(cost - best_tl)
        if idx // 10 > 0 and idx % 10 == 0:
            #print(round(loss.item(), 4), round(tl.item(), 4), num_P, round(opt_gap.item(), 4))
            train_curve[j_idx][idx // 10] = loss.item()
            hard_train_curve[j_idx][idx // 10] = tl
            hashmap_size[j_idx][idx // 10] = (len(hashmap), num_P, sum_thresh)
            
        if "pgd" in alg:
            for param in model.parameters():
                row, col = linear_sum_assignment(param.grad.data.cpu())
                P = torch.zeros_like(W)
                for pos_idx in range(len(row)):
                   i = row[pos_idx]
                   j = col[pos_idx]
                   P[i, j] = 1
                param.data = (1 - lr)*param.data + lr*P
            torch.nn.Module.zero_grad(model)
            
        else:
            optimizer.step()
            optimizer.zero_grad()

        if opt_gap <= 0.001 or patience <= 0:
            print(cost, best_tl)#, patience)
            break

np.save(f"train_be/train_curve_{lr}_{num_terms}_{alg}_{S}_markov.npy", train_curve)
np.save(f"train_be/hard_train_curve_{lr}_{num_terms}_{alg}_{S}_markov.npy", hard_train_curve)
np.save(f"train_be/hashmap_size_{lr}_{num_terms}_{alg}_{S}_markov.npy", hashmap_size)
