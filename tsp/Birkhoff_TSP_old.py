import torch
from torch.nn.functional import softmax
import numpy as np
import torch.nn as nn
from qap.IPF import *
#from dimsum import *
import matplotlib.pyplot as plt
import networkx as nx
import scipy
import tqdm
import hashlib
import time
import random
import scipy.spatial.distance as distance
from scipy.sparse.csgraph import maximum_bipartite_matching
torch.set_printoptions(sci_mode = False , precision = 3)
np.set_printoptions(precision=3, suppress = True)


######### Weights ##############
#Functions that compute weights to be used in generating a 
#Birkhoff decomposition
def get_weights_Markov(points):
    n = len(points) - 1
    M = torch.zeros(n,n)
    for i in range(n):
        for j in range(n):
            M[i,j] = 1-l2(points[i], points[j])/l2diam(points)

    #M = M*(1 - torch.eye(n))
    M = ipf(M, n^2)

    P = torch.zeros(n,n)
    for i in range(n):
        if i == 0:
            for j in range(n):
                P[i,j] = M[i,j]
        else:
             for j in range(n):
                P[i,j] = sum( P[i-1,k]*M[k,j] for k in range(n))
    return P

def get_walk(points, A):
    n = len(points)
    walk = [n-1]
    while len(walk) < n:
        r = np.random.rand()*sum(A[walk[-1],i] for i in range(n) if not i in walk)
        tot = 0
        found = False
        for i in range(n):
            if not i in walk and found == False:
                tot += A[walk[-1],i]
                if r <= tot:
                    walk += [i]
                    found = True
    return walk
    
def get_weights_sample(points, m):
    n = len(points)
    M = torch.zeros(n,n)
    for i in range(n):
        for j in range(n):
            M[i,j] = 1-l2(points[i], points[j])/l2diam(points)
    M = M*(1 - torch.eye(n))
    A = ipf(M, n^2)

    walks = []
    for i in range(m):
        walks += [get_walk(points, A)]
    
    B = np.zeros((n-1, n-1))
    for i in range(m):
        for k in range(len(walks[i])-1):
            B[k,walks[i][k+1]] += 1
    B = B/B.max()
    return B + .001*np.random.rand(B.shape[0],B.shape[1])
    
    
def objective_function(P, D, n, fast=True):
    if fast:
        # Fast parallel CUDA implementation
        # Create all pairs at once
        P_curr = P
        P_next = torch.roll(P, -1, dims=1)
        
        # Compute all matrix multiplications in parallel
        # P_curr.T @ D @ P_next
        intermediate = torch.matmul(D, P_next)
        obj = torch.sum(P_curr * intermediate)
        
        return obj
    else:
        # Original implementation
        obj = 0
        for i in range(n-1):
            obj += torch.matmul(torch.matmul(P[:, i], D), P[:, i+1])
        obj += torch.matmul(torch.matmul(P[:, i+1], D), P[:, 0])
        return obj

######## Initializing ###########

def generate_points(n):
    V = torch.rand([n,2])
    return V
def get_l1_dist(V):
    dist =  torch.tensor([ [l1(x,y)  for x in V] for y in V ])
    return dist
def get_l2_dist(V):
    dist =  torch.tensor([ [l2(x,y)  for x in V] for y in V ])
    return dist



######## Utility ###########

def l2(a,b):
    return torch.sqrt((a[0] - b[0])**2  + (a[1] - b[1])**2)

def l1(a,b):
    return torch.abs(a[0] - b[0]) + torch.abs(a[1] - b[1])

def l2diam(points):
    return torch.max(get_l2_dist(points))

def tour_length(T, points):
    length = 0
    for i in range(len(T)):
        length += l2(points[T[i-1]], points[T[i]])
    return length 

def tour_plot(T, points):
    points = np.array(points)
    T_n = torch.cat([T,  torch.tensor([len(points)-1])])
    plt.figure()
    plt.scatter(points[:,0], points[:,1])
    for i in range(len(points)):
        plt.arrow(points[T_n[i-1]][0], points[T_n[i-1]][1], points[T_n[i]][0] - points[T_n[i-1]][0], points[T_n[i]][1] - points[T_n[i-1]][1])
    plt.show()

class MatrixModel_qp(nn.Module):
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
    
######## Continuous Birkhoff 
def cont_Birkhoff_SFE(W, k, D, perms, setting, noise=False, device="cpu"):
    n = W.shape[0]
    fill = -n
    min_tl = n
    new_W = W.clone()
    cap = setting[1]
    min_P = None
    sum_thresh = 0
    stop = False
    to_choice = None
    total_loss = 0
    
    # Store all permutations
    all_permutations = []
    
    # if noise:
    #     perms = 0.8*perms + 0.2*torch.rand(n, n)
    
    for idx in range(k):
        A = torch.where(new_W > 0, perms, fill)
        A_to_use = A.detach().cpu().numpy()
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(A_to_use, maximize=True)
        thresh = torch.amin(new_W[row_ind, col_ind])
        sum_thresh += thresh
        P = torch.zeros(n, n)
        for q in range(len(row_ind)):
            i = row_ind[q]
            j = col_ind[q]
            P[i,j] = 1
            new_W[i,j] = new_W[i,j] - thresh
        
        # Store the permutation
        all_permutations.append(P)
        
        tl = objective_function(P, D, n)
        
        if tl < min_tl:
            min_tl = tl
            min_P = P
        
        total_loss += tl*thresh
    
    total_loss = total_loss/sum_thresh
    return min_tl, total_loss, min_P, idx, sum_thresh.item(), all_permutations

def cont_Birkhoff_decomp(W, k, points, perm, hashmap, alg="gd"):
    #returns a Birhoff decomp using perm to set weights 
    #Each element of the returned list is (alpha, P_alpha) 
    #where alpha is the Birkhoff coefficient and P_alpha is 
    # the corresponding permuation matrix
    decomp = []
    for i in range(k):
        step = step_cont_assign(W, points, perm, p_perm, fill, hashmap, return_tour=True)
        W = step[0]
        decomp += [(step[1],step[2])]
    return decomp 


def step_cont_assign(W, D, p_perm, fill, hashmap, return_tour=False):
    #Performs one step in the Birkhoff decomp
    n = W.size()[0]
    fill = -n
    A = torch.where(W > 0, p_perm, fill).numpy()
    #A = np.zeros(W.shape)
    #for i in range(B.shape[0]):
    #    for j in range(B.shape[1]):
    #        if B[i,j] > 0:
    #            A[i,j] = 2**perm[i,j]
    #        else:
    #            A[i,j] = 2*(W.size()[0]*W.size()[1])
    hash_value = hashlib.sha256(A.tobytes()).hexdigest()
    if hash_value in hashmap:
        hashmap.move_to_end(hash_value)
        row_ind, col_ind = hashmap[hash_value]
    else:
        maximum_bipartite_matching
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(A, maximize=True)
        hashmap[hash_value] = (row_ind, col_ind)
    thresh = torch.amin(W[row_ind, col_ind])
    W_new = W.clone()
    P = torch.zeros_like(W)
    #prev_j = col_ind[-1]
    #tl = 0
    for q in range(len(row_ind)):
        i = row_ind[q]
        j = col_ind[q]
        P[i,j] = 1
        W_new[i,j] = W_new[i,j] - thresh  
    
    #print(row_ind, col_ind)
    #print(A)
    #print(tl)
    #print(W[row_ind, col_ind])
    #print(thresh)
    tl = objective_function(P, D, n)
    return W_new, thresh, tl 

def cont_optimal_Birkhoff_tour(W,k, points,perm, hashmap):
    #Recovers optimal tour from W
    decomp = cont_Birkhoff_decomp(W, k,points,perm, hashmap)
    #print(f'Birkhoff decomp error:{ 1 - sum(d[0] for d in decomp)}')
    decomp.sort(key=lambda a : tour_length(a[1], points))
    return  decomp[0][1]
    
        
######## Model ###########


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")



class MatrixModel(nn.Module):
    def __init__(self, num_terms, points):
        super().__init__()
        self.num_terms = num_terms - 1
        self.flatten = nn.Flatten()
        self.mat =  nn.Linear( self.num_terms,self.num_terms   )
        self.points = points
        #self.perms = [get_weights_sample(points, 100000)]
        self.perms = [get_weights_Markov(points)]
        self.k = self.num_terms*self.num_terms - self.num_terms

    def forward(self):
        W = torch.abs(self.mat.weight)
        W = ipf(W, self.num_terms**2, 1,1)
        return (sum([cont_Birkhoff_SFE(W, self.k, self.points,self.perms[i])[0] for i in range(len(self.perms))])/len(self.perms), cont_Birkhoff_SFE(W, self.k, self.points,self.perms[0])[1])
    

####### Training ###########


def train( model, optimizer):

    model.train()
    # Compute prediction error
    loss = model.forward()[0]
    print("Loss:")
    print(loss)
    # Backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss


######### Main #########


def main():
    num_terms = 10
    epochs = 10
    points = generate_points(num_terms)
    print(f'points:{points}')

    model = MatrixModel( num_terms, points).to(device)
    #print(model.perm)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
    data = []
    for t in tqdm.tqdm(range(epochs)):
        print(f"Epoch {t+1}\n-------------------------------")
        data += [train(model, optimizer).cpu().detach()]

    loss, W= model.forward()

    T = cont_optimal_Birkhoff_tour(W,model.k,points, model.perms[0])
    print(f'point set size: {len(points)}')
    print(f'final soft loss:{loss}')
    print(f'gradient hard loss: {tour_length(T, points)}')
    tour_plot(T, points)
    plt.plot(data)
    plt.show()
    plt.figure()
    
#main()

