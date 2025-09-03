import torch
import numpy as np
import torch.nn as nn
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

from Birkhoff_TSP import *

import sys
import logging
import math
import random
from collections import OrderedDict
from scipy.optimize import linear_sum_assignment
import sys

device = torch.device("cuda:6")

S = str(sys.argv[2]) 
alg = str(sys.argv[1])
lr = float(sys.argv[3])
#target = int(sys.argv[3])
print(sys.argv)
T = 2000
target_num_terms = 10
num_terms = target_num_terms**2
target_num_input = 100
input_lst = []
#hashmap = OrderedDict()
#train_curve = []
#val_curve = []
data_lst = torch.load("data_lst_10.pt")
perm_lst = torch.load("perm_lst_10.pt")

if S == "predict":
    gen_perm_model.load_state_dict(torch.load("predict_s.model"))


print(len(data_lst), len(perm_lst))
idx = 0
result_lst = []
stop = False

W_init_lst = []

while len(input_lst) < target_num_input and not stop:
    points = data_lst[idx]
    W_init_lst.append(perm_lst[idx])
    num_terms = points.shape[0]

    if num_terms == target_num_terms:
        if S == 'rand':                          
            perms = torch.rand((num_terms, num_terms))
	 
        elif S == 'l1':
            perms = get_l1_dist(points)
         
        elif S == 'l2':
            perms = get_l2_dist(points)

        elif S == 'l2_perm':
            perms = torch.Tensor(np.random.RandomState(seed=42).permutation(get_l2_dist(points)))

        elif S == 'predict':
            n = num_terms
            perms = gen_perm_model(points)
            perms = perms.detach()
            D = get_l2_dist(points)
            result_lst.append(objective_function(perms.numpy(), D, n))

        input_lst.append((points, perms))
    idx += 1
idx = 0
train_curve = [[] for i in range(target_num_input)]
hard_train_curve = [[] for i in range(target_num_input)]

def objective_function(P, D, n):
    obj = 0
    for i in range(n-1):
        obj += torch.matmul(torch.matmul(P[:, i], D), P[:, i+1])
    return obj
    
W_init_lst = torch.load(f"W_{num_terms}_mst.pt")

print(len(W_init_lst))

class MatrixModel(nn.Module):
    def __init__(self, num_terms, algo="gd"):
        super().__init__()
        self.num_terms = num_terms
        self.mat = nn.Linear(self.num_terms,self.num_terms, bias=False) 
        self.algo = algo
    def forward(self):
        if self.algo == "gd":
            W = torch.abs(self.mat.weight)
            W = ipf(W, self.num_terms, 1,1)
        else:
            W = self.mat.weight
        return W
        
for j_idx in tqdm.tqdm(range(target_num_input)):
    W_init = W_init_lst[j_idx]
    model = MatrixModel(num_terms, algo=alg).to(device)
    state_dict = model.state_dict()
    state_dict['mat.weight'] = W_init
    model.load_state_dict(state_dict)
    points, D = input_lst[j_idx]
    points = points.to(device)
    D = D.to(device)
    
    for idx in range(T):
        W = model.forward()
        loss = objective_function(W, D, points.shape[0])
        loss.backward()
        
        for param in model.parameters():
            row, col = linear_sum_assignment(param.grad.data.cpu())
            P = torch.zeros_like(W)
            #tour = torch.zeros(num_terms, dtype = int)
            for q_idx in range(len(row)):
                i = row[q_idx]
                j = col[q_idx]
                P[i, j] = 1
            #P = torch.Tensor(min_product(param.grad.data)) 
            param.data = (1 - lr)*param.data + lr*P
    
        #torch.nn.Module.zero_grad(model)
        #tl = tour_length(tour, points).cpu().item()
        #hard_train_curve[j_idx].append(tl)
        #print(loss.item(), tl)
        
        if idx >= 10 and idx % 10 == 0:
            train_curve[j_idx].append(loss.detach().cpu().item())
            W_clone = W.clone().detach().cpu()
            row_ind, col_ind = linear_sum_assignment(W_clone, maximize=True)
            tour = torch.zeros(num_terms, dtype = int)
            for q in range(len(row_ind)):
                i = row_ind[q]
                j = col_ind[q]
                tour[i] = j
            tl = tour_length(tour, points).cpu().item()
            hard_train_curve[j_idx].append(tl)
            print(loss.item(), tl)

np.save(f"train/train_curve_{lr}_{num_terms}_{alg}_{S}.npy", train_curve)
np.save(f"train/hard_train_curve_{lr}_{num_terms}_{alg}_{S}.npy", hard_train_curve)
