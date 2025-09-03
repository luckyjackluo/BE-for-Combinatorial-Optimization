import torch
from torch.nn.functional import softmax
import numpy as np
import torch.nn as nn
from IPF import *
import matplotlib.pyplot as plt
import networkx as nx
import scipy
import tqdm
import hashlib
import time
import scipy.spatial.distance as distance
from scipy.sparse.csgraph import maximum_bipartite_matching
torch.set_printoptions(sci_mode = False , precision = 3)
np.set_printoptions(precision=3, suppress = True)


######### Adjacency Matrix TSP Functions ##############

def adjacency_to_tour_greedy(adj_matrix, points):
    """
    Convert adjacency matrix to tour using greedy decoding.
    Args:
        adj_matrix: NxN adjacency matrix (edge scores)
        points: Nx2 array of coordinates
    Returns:
        tour: List of indices representing the tour
    """
    N = len(points)
    
    # Handle edge cases
    if np.isnan(adj_matrix).any() or np.isinf(adj_matrix).any():
        return list(range(N))  # Return simple sequential tour as fallback
    
    # Calculate normalized scores (Aij + Aji)/(||ci - cj||)
    scores = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                dist = np.linalg.norm(points[i] - points[j])
                if dist == 0:
                    scores[i,j] = 0  # Same point, no edge
                else:
                    scores[i,j] = (adj_matrix[i,j] + adj_matrix[j,i]) / dist
    
    # Initialize tour with first city
    tour = [0]
    used = {0}
    
    # Greedily build tour
    while len(tour) < N:
        current = tour[-1]
        # Get scores for unused cities from current city
        candidates = [(scores[current,j], j) for j in range(N) if j not in used]
        if not candidates:
            # This shouldn't happen, but just in case
            remaining = [j for j in range(N) if j not in used]
            if remaining:
                tour.append(remaining[0])
                used.add(remaining[0])
            break
        # Choose highest scoring unused city
        _, next_city = max(candidates)
        tour.append(next_city)
        used.add(next_city)
    
    return tour

def adjacency_to_tour_direct(adj_matrix, threshold=0.5):
    """
    Convert adjacency matrix to tour by following edges directly.
    Args:
        adj_matrix: NxN adjacency matrix (edge weights)
        threshold: Minimum edge weight to consider
    Returns:
        tour: List of indices representing the tour, or None if no valid tour
    """
    N = adj_matrix.shape[0]
    
    # Create binary adjacency matrix based on threshold
    binary_adj = (adj_matrix > threshold).astype(int)
    
    # Check if each node has exactly 2 edges (valid TSP tour)
    degrees = np.sum(binary_adj, axis=1)
    if not all(deg == 2 for deg in degrees):
        return None  # Not a valid tour
    
    # Follow the tour starting from node 0
    tour = [0]
    visited = {0}
    current = 0
    
    while len(tour) < N:
        # Find next unvisited neighbor
        neighbors = np.where(binary_adj[current] == 1)[0]
        next_node = None
        for neighbor in neighbors:
            if neighbor not in visited:
                next_node = neighbor
                break
        
        if next_node is None:
            return None  # No valid continuation
        
        tour.append(next_node)
        visited.add(next_node)
        current = next_node
    
    # Check if tour closes properly
    if binary_adj[current, 0] == 1:
        return tour
    else:
        return None

def objective_function_adjacency(adj_matrix, points, decode_method='greedy'):
    """
    Compute TSP objective for adjacency matrix.
    Args:
        adj_matrix: NxN adjacency matrix
        points: Nx2 coordinates
        decode_method: 'greedy' or 'direct'
    Returns:
        tour_length: Total length of the tour
    """
    if isinstance(adj_matrix, torch.Tensor):
        adj_np = adj_matrix.detach().cpu().numpy()
        points_np = points.detach().cpu().numpy() if isinstance(points, torch.Tensor) else points
    else:
        adj_np = adj_matrix
        points_np = points
    
    if decode_method == 'direct':
        tour = adjacency_to_tour_direct(adj_np)
        if tour is None:
            # Fallback to greedy if direct method fails
            tour = adjacency_to_tour_greedy(adj_np, points_np)
    else:
        tour = adjacency_to_tour_greedy(adj_np, points_np)
    
    # Compute tour length
    length = 0.0
    for i in range(len(tour)):
        a = points_np[tour[i-1]]
        b = points_np[tour[i]]
        length += np.linalg.norm(a - b)
    
    return length

def objective_function_permutation_fast(perm_matrix, points, distance_matrix=None):
    """
    Fast TSP objective computation for pure permutation matrices.
    Since P_merged is a permutation matrix, we can directly compute tour length
    without decoding to tour first.
    
    Args:
        perm_matrix: NxN permutation matrix (torch.Tensor)
        points: Nx2 coordinates (torch.Tensor)
        distance_matrix: Pre-computed distance matrix (optional, for efficiency)
    Returns:
        tour_length: Total length of the tour (float)
    """
    if isinstance(perm_matrix, torch.Tensor):
        P = perm_matrix
    else:
        P = torch.tensor(perm_matrix, dtype=torch.float32)
    
    # Use pre-computed distance matrix if available, otherwise compute it
    if distance_matrix is not None:
        D = distance_matrix
    else:
        # Create distance matrix using vectorized operations
        pts = points if isinstance(points, torch.Tensor) else torch.tensor(points, dtype=torch.float32)
        n = P.shape[0]
        # Vectorized distance computation
        pts_expanded_i = pts.unsqueeze(1).expand(n, n, 2)  # [n, n, 2]
        pts_expanded_j = pts.unsqueeze(0).expand(n, n, 2)  # [n, n, 2]
        D = torch.norm(pts_expanded_i - pts_expanded_j, dim=2)  # [n, n]
    
    # Tour length = sum of P[i,j] * D[i,j] for all i,j
    # Since P is binary, this gives us exactly the edges in the tour
    tour_length = torch.sum(P * D).item()
    
    return tour_length

######## Utility Functions (same as original) ###########

def l2(a,b):
    return torch.sqrt((a[0] - b[0])**2  + (a[1] - b[1])**2)

def l1(a,b):
    return torch.abs(a[0] - b[0]) + torch.abs(a[1] - b[1])

def get_l2_dist(V):
    dist =  torch.tensor([ [l2(x,y)  for x in V] for y in V ])
    return dist

def l2diam(points):
    return torch.max(get_l2_dist(points))

def tour_length(T, points):
    length = 0
    for i in range(len(T)):
        length += l2(points[T[i-1]], points[T[i]])
    return length 

######## Permutation Matrix Merging Functions ########

def matrix_to_perm_list(P):
    """Convert permutation matrix to list representation."""
    return [int(np.where(row == 1)[0][0]) for row in P]

def extract_cycles(perm):
    """Extract disjoint cycles from permutation list."""
    N = len(perm)
    visited = [False] * N
    cycles = []

    for i in range(N):
        if not visited[i]:
            cycle = []
            current = i
            while not visited[current]:
                visited[current] = True
                cycle.append(current)
                current = perm[current]
            cycles.append(cycle)
    return cycles

def merge_cycles(cycles):
    """Greedily merge all cycles into one."""
    # Flatten cycles by linking end of one to start of next
    merged = []
    for i in range(len(cycles)):
        merged += cycles[i]
    return merged + [merged[0]]  # make it a full tour

def tour_to_perm_matrix(tour):
    """Convert a single tour to permutation matrix."""
    N = len(tour) - 1  # last element is same as first
    P_new = np.zeros((N, N), dtype=int)
    for i in range(N):
        P_new[tour[i], tour[i+1]] = 1
    return P_new

def merge_permutation_matrix(P):
    """Merge multiple subtours in permutation matrix into single tour."""
    if isinstance(P, torch.Tensor):
        P_np = P.detach().cpu().numpy()
    else:
        P_np = P
    
    perm = matrix_to_perm_list(P_np)
    cycles = extract_cycles(perm)
    merged_tour = merge_cycles(cycles)
    P_new = tour_to_perm_matrix(merged_tour)
    return torch.tensor(P_new, dtype=torch.float32) if isinstance(P, torch.Tensor) else P_new

def permutation_to_adjacency(P):
    """
    Convert permutation matrix directly to symmetric adjacency matrix.
    This treats the permutation as defining directed edges and makes them undirected.
    """
    if isinstance(P, torch.Tensor):
        P_np = P.detach().cpu().numpy()
        is_torch = True
    else:
        P_np = P
        is_torch = False
    
    # Make symmetric adjacency matrix (undirected graph)
    A = P_np + P_np.T
    # Remove any values > 1 (where both directions exist)
    A = np.minimum(A, 1.0)
    
    return torch.tensor(A, dtype=torch.float32) if is_torch else A

######## Modified Birkhoff Decomposition for Adjacency Matrices ########

def cont_Birkhoff_SFE_adjacency(W, k, points, perms, hashmap, setting, alg, distance_matrix=None, m=True, device="cpu"):
    """
    Continuous Birkhoff decomposition for adjacency matrices.
    Now treats each permutation matrix as an adjacency matrix for TSP evaluation.
    Optimized version based on QAP implementation.
    """
    n = W.size()[0]
    fill = -n
    new_W = W.clone().to(device)
    min_tl = float('inf')
    cap = setting[1]
    min_tour = None
    best_perm = None  # Track the best permutation matrix
    sum_thresh = 0
    perm_loss = 0
    
    # Pre-allocate storage for efficiency
    all_threshs = []
    all_tl = []
    
    # Pre-allocate P matrix and reuse it
    P = torch.zeros(n, n, device=device)
    
    # Use provided distance matrix or compute it if not available
    if distance_matrix is not None:
        dist_matrix = distance_matrix
    else:
        # Pre-compute distance matrix once for efficiency
        pts = points if isinstance(points, torch.Tensor) else torch.tensor(points, dtype=torch.float32)
        pts_expanded_i = pts.unsqueeze(1).expand(n, n, 2)  # [n, n, 2]
        pts_expanded_j = pts.unsqueeze(0).expand(n, n, 2)  # [n, n, 2]
        dist_matrix = torch.norm(pts_expanded_i - pts_expanded_j, dim=2)  # [n, n]
    
    if "noise" in alg:
        perms = 0.9*perms + 0.1*torch.rand(n, n)
    
    for idx in range(k):
        # Compute assignment matrix efficiently
        A_cpu = torch.where(new_W > 0, perms, fill).detach().cpu().numpy()
        hash_value = hashlib.sha256(A_cpu.tobytes()).hexdigest()
        
        if hash_value in hashmap:
            row_ind, col_ind = hashmap[hash_value]
        else:
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(A_cpu, maximize=m)
            hashmap[hash_value] = (row_ind, col_ind) 
        
        # Get threshold and update sum in one operation
        thresh = torch.amin(new_W[row_ind, col_ind]) 
        sum_thresh += thresh
        all_threshs.append(thresh.item())
        
        # Create permutation matrix efficiently - reset and fill
        P.zero_()  # Reset P matrix
        P[row_ind, col_ind] = 1
        
        # Update new_W efficiently
        new_W[row_ind, col_ind] -= thresh
        
        # Merge any subtours in the permutation matrix into a single tour
        P_merged = merge_permutation_matrix(P)
        
        # Evaluate merged P as permutation matrix for TSP (fast method with pre-computed distances)
        tl = objective_function_permutation_fast(P_merged, points, dist_matrix)
        all_tl.append(tl)
        
        # Update minimum and best permutation
        if tl < min_tl:
            min_tl = tl
            min_tour = row_ind
            best_perm = P_merged
        elif best_perm is None:
            # Initialize best_perm with first permutation if none found yet
            best_perm = P_merged
        
        if idx >= cap:
            break
    
    # Compute weighted loss efficiently
    total_loss = 0
    for i, (tl, thresh) in enumerate(zip(all_tl, all_threshs)):
        total_loss += tl * thresh
    
    # Convert min_tl to tensor for consistency
    min_tl_tensor = torch.tensor(min_tl, dtype=torch.float32)
    
    return total_loss/sum_thresh, perm_loss, min_tl_tensor, sum_thresh, col_ind, best_perm

######## Initialization Functions (same as original) ###########

def generate_points(n):
    V = torch.rand([n,2])
    return V

def get_l1_dist(V):
    dist =  torch.tensor([ [l1(x,y)  for x in V] for y in V ])
    return dist

######## Weights/Permutation Initialization ###########

def get_weights_Markov(points):
    n = len(points) - 1
    M = torch.zeros(n,n)
    for i in range(n):
        for j in range(n):
            M[i,j] = 1-l2(points[i], points[j])/l2diam(points)

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

def get_adjacency_init(points):
    """
    Initialize adjacency matrix based on distance (closer points have higher edge weights).
    """
    n = len(points)
    D = get_l2_dist(points)
    max_dist = torch.max(D)
    # Invert distances to get edge weights (closer = higher weight)
    A = (max_dist - D) / max_dist
    # Remove self-loops
    A = A * (1 - torch.eye(n))
    return A

def tour_plot(T, points):
    points = np.array(points)
    T_n = T
    plt.figure()
    plt.scatter(points[:,0], points[:,1])   
    for idx in range(len(points)):
        plt.annotate(idx, (points[idx, 0], points[idx, 1]))
    for i in range(len(points)):
        plt.arrow(points[T_n[i-1]][0], points[T_n[i-1]][1], points[T_n[i]][0] - points[T_n[i-1]][0], points[T_n[i]][1] - points[T_n[i-1]][1])
    plt.show() 