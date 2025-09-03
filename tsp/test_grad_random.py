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

S = str(sys.argv[2]) 
alg = str(sys.argv[1])
lr = float(sys.argv[3])
target_num_terms = int(sys.argv[4])
#device = torch.device(f"cuda:{device_idx}")
device = torch.device(f"cpu")
print(sys.argv)
T = 10000
# target_num_terms = 15
num_terms = target_num_terms
target_num_input = 20
input_lst = []
#hashmap = OrderedDict()
#train_curve = []
#val_curve = []
#if "random" in S:
data_lst = torch.load(f"input_data/data_lst_{target_num_terms}_random.pt")
#perm_lst = torch.load(f"input_data/perm_lst_{target_num_terms}.pt")
dist_lst = torch.load(f"input_data/dist_lst_{target_num_terms}_random.pt")

if S == "predict":
    gen_perm_model.load_state_dict(torch.load("predict_s.model"))

#print(len(data_lst), len(perm_lst))
idx = 0
result_lst = []
stop = False

W_init_lst = []

while len(input_lst) < target_num_input and not stop:
    points = data_lst[idx] #perm_lst[idx]
    num_terms = points.shape[0]
    #perms = ipf(perms, 10)

    if num_terms == target_num_terms:
        D = dist_lst[idx]
        #D = get_l2_dist(points)
        if S == 'random':
            perms = torch.rand(num_terms, num_terms)
        elif S == 'predict':
            n = num_terms
            perms = gen_perm_model(points)
            perms = perms.detach()
            result_lst.append(objective_function(perms.numpy(), D, n))

        W_init_lst.append(perms)
        input_lst.append((points, D))
        
    idx += 1
idx = 0
train_curve = [[] for i in range(target_num_input)]
hard_train_curve = [[] for i in range(target_num_input)]

def objective_function(P, D, n):
    obj = 0
    for i in range(n-1):
        obj += torch.matmul(torch.matmul(P[:, i], D), P[:, i+1])
    obj += torch.matmul(torch.matmul(P[:, i+1], D), P[:, 0])
    return obj

if S == "mst":
    W_init_lst = torch.load(f"W_{num_terms}_mst.pt")
if S == "qp":
    W_init_lst = torch.Tensor(np.load(f"train/final_sol_0.01_{num_terms}_pgd_markov.npy")[:, 0])

class MatrixModel(nn.Module):
    def __init__(self, num_terms, algo="gd"):
        super().__init__()
        self.num_terms = num_terms
        self.mat = nn.Linear(self.num_terms,self.num_terms, bias=False) 
        self.algo = algo
    def forward(self):
        if self.algo == "gd":
            W = torch.abs(self.mat.weight)
            W = ipf(W, self.num_terms*2, 1,1)
        else:
            W = self.mat.weight
        return W
        

final_sol = []
for j_idx in tqdm.tqdm(range(target_num_input)):
    W_init = W_init_lst[j_idx]
    model = MatrixModel(num_terms, algo=alg).to(device)
    state_dict = model.state_dict()
    state_dict['mat.weight'] = ipf(torch.abs(W_init), 5, 1,1)
    model.load_state_dict(state_dict)
    points, D = input_lst[j_idx]
    points = points.to(device)
    D = D.to(device)

    for idx in range(T):
        W = model.forward()
        loss = objective_function(W, D, points.shape[0])
        loss.backward()
        if alg == "pgd":
            for param in model.parameters():
                row, col = linear_sum_assignment(param.grad.data.cpu())
                P = torch.zeros_like(W)
                for q_idx in range(len(row)):
                    i = row[q_idx]
                    j = col[q_idx]
                    P[i, j] = 1
                #P = torch.Tensor(min_product(param.grad.data))
                param.data = (1 - lr)*param.data + lr*P
        else:
            for param in model.parameters():
                param.data = param.data - lr*param.grad.data

        torch.nn.Module.zero_grad(model)
        
        if idx >= 10 and idx % 10 == 0:
            train_curve[j_idx].append(loss.detach().cpu().item())
            W_clone = W.clone().detach().cpu()
            row_ind, col_ind = linear_sum_assignment(W_clone, maximize=True)
            P_round = torch.zeros_like(W_clone)
            for q in range(len(row_ind)):
                i = row_ind[q]
                j = col_ind[q]
                P_round[i, j] = 1
            tl = objective_function(P_round, D, points.shape[0]).item()
            hard_train_curve[j_idx].append(tl)
            #print(loss.item(), tl)
     
    final_sol.append([W_clone, P_round])

np.save(f"train/train_curve_{lr}_{num_terms}_{alg}_{S}_random.npy", train_curve)
np.save(f"train/hard_train_curve_{lr}_{num_terms}_{alg}_{S}_random.npy", hard_train_curve)
np.save(f"train/final_sol_{lr}_{num_terms}_{alg}_{S}_random.npy", final_sol)
