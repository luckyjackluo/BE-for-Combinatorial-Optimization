import numpy as np
from gurobi_tsp import *
from Birkhoff_TSP import *
import sys
import logging
import math
import random
import networkx
from collections import defaultdict
from itertools import combinations
from tqdm import tqdm

seed = 1
random.seed(seed)
epochs = 300
data_lst = []
tour_lst = []
cost_lst = []
dist_lst = []
output_tour = []
output_tl = []
mst_perms = []
npoints = 10
num_terms = npoints

for t in tqdm(range(epochs)):
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

    dist_lst.append(dist)
    nodes = list(range(npoints))
    tour, cost = solve_tsp(nodes, distances)
    data_lst.append(data)
    tour_lst.append(tour)
    cost_lst.append(cost)

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
    tl = objective_function(P, distances, num_terms)
    init_perm = torch.zeros((num_terms, num_terms))
    for pos in range(len(tour)):
        city_idx = tour[pos]
        init_perm[city_idx][pos] = 1
    output_tour.append(tour)
    output_tl.append(tl.item())
    mst_perms.append(init_perm)

torch.save(data_lst, f"input_data/data_lst_{npoints}.pt")
torch.save(tour_lst, f"input_data/tour_lst_{npoints}.pt")
torch.save(dist_lst, f"input_data/dist_lst_{npoints}.pt")
torch.save(cost_lst, f"input_data/cost_lst_{npoints}.pt")
torch.save(mst_perms, f"input_data/W_{npoints}_mst.pt")
torch.save(output_tl, f"input_data/cost_lst_{npoints}_mst.pt")
torch.save(output_tour, f"input_data/tour_lst_{npoints}_mst.pt")
