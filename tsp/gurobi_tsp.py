import sys
import logging
import math
import random
from collections import defaultdict
from itertools import combinations

import gurobipy as gp
from gurobipy import GRB


def shortest_subtour(edges):
    """Given a list of edges, return the shortest subtour (as a list of nodes)
    found by following those edges. It is assumed there is exactly one 'in'
    edge and one 'out' edge for every node represented in the edge list."""

    # Create a mapping from each node to its neighbours
    node_neighbors = defaultdict(list)
    for i, j in edges:
        node_neighbors[i].append(j)
    assert all(len(neighbors) == 2 for neighbors in node_neighbors.values())

    # Follow edges to find cycles. Each time a new cycle is found, keep track
    # of the shortest cycle found so far and restart from an unvisited node.
    unvisited = set(node_neighbors)
    shortest = None
    while unvisited:
        cycle = []
        neighbors = list(unvisited)
        while neighbors:
            current = neighbors.pop()
            cycle.append(current)
            unvisited.remove(current)
            neighbors = [j for j in node_neighbors[current] if j in unvisited]
        if shortest is None or len(cycle) < len(shortest):
            shortest = cycle

    assert shortest is not None
    return shortest


class TSPCallback:
    """Callback class implementing lazy constraints for the TSP.  At MIPSOL
    callbacks, solutions are checked for subtours and subtour elimination
    constraints are added if needed."""

    def __init__(self, nodes, x):
        self.nodes = nodes
        self.x = x

    def __call__(self, model, where):
        """Callback entry point: call lazy constraints routine when new
        solutions are found. Stop the optimization if there is an exception in
        user code."""
        if where == GRB.Callback.MIPSOL:
            try:
                self.eliminate_subtours(model)
            except Exception:
                logging.exception("Exception occurred in MIPSOL callback")
                model.terminate()

    def eliminate_subtours(self, model):
        """Extract the current solution, check for subtours, and formulate lazy
        constraints to cut off the current solution if subtours are found.
        Assumes we are at MIPSOL."""
        values = model.cbGetSolution(self.x)
        edges = [(i, j) for (i, j), v in values.items() if v > 0.5]
        tour = shortest_subtour(edges)
        if len(tour) < len(self.nodes):
            # add subtour elimination constraint for every pair of cities in tour
            model.cbLazy(
                gp.quicksum(self.x[i, j] for i, j in combinations(tour, 2))
                <= len(tour) - 1
            )


def solve_tsp(nodes, distances):
    """
    Solve a dense symmetric TSP using the following base formulation:

    min  sum_ij d_ij x_ij
    s.t. sum_j x_ij == 2   forall i in V
         x_ij binary       forall (i,j) in E

    and subtours eliminated using lazy constraints.
    """
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
    
        with gp.Model(env=env) as m:
            # Create variables, and add symmetric keys to the resulting dictionary
            # 'x', such that (i, j) and (j, i) refer to the same variable.
            x = m.addVars(distances.keys(), obj=distances, vtype=GRB.BINARY, name="e")
            x.update({(j, i): v for (i, j), v in x.items()})
    
            # Create degree 2 constraints
            for i in nodes:
                m.addConstr(gp.quicksum(x[i, j] for j in nodes if i != j) == 2)
    
            # Optimize model using lazy constraints to eliminate subtours
            m.Params.LazyConstraints = 1
            cb = TSPCallback(nodes, x)
            m.optimize(cb)
    
            # Extract the solution as a tour
            edges = [(i, j) for (i, j), v in x.items() if v.X > 0.5]
            tour = shortest_subtour(edges)
            assert set(tour) == set(nodes)
    
            return tour, m.ObjVal
