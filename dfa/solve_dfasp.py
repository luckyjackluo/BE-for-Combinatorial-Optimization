import argparse
import csv
import os
import time
from pathlib import Path
from typing import Tuple

import networkx as nx
import gurobipy as gp
from gurobipy import GRB
from tqdm import tqdm


def solve_dfasp(G: nx.DiGraph, silent: bool = True, time_limit: float | None = None) -> Tuple[int, float]:
    """Solve DFASP on a given directed graph using Gurobi.

    Parameters
    ----------
    G : nx.DiGraph
        Input directed graph.
    silent : bool, optional
        If True, turn off Gurobi console output.
    time_limit : float | None, optional
        Time limit for Gurobi in seconds (None for no limit).

    Returns
    -------
    obj : int
        Optimal objective value – size of the minimum feedback arc set.
    runtime : float
        Wall-clock runtime reported by Gurobi (seconds).
    """
    n = G.number_of_nodes()

    # Set Gurobi license file
    license_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "gurobi.lic")
    if not os.path.exists(license_path):
        raise FileNotFoundError(f"Gurobi license file not found at {license_path}")
    os.environ["GRB_LICENSE_FILE"] = license_path

    model = gp.Model()
    if silent:
        model.Params.LogToConsole = 0

    # Apply time limit if provided (>0 seconds)
    if time_limit is not None and time_limit > 0:
        model.Params.TimeLimit = time_limit

    # Decision variables
    x = model.addVars(G.edges(), vtype=GRB.BINARY, name="x")
    y = model.addVars(G.nodes(), vtype=GRB.INTEGER, lb=1, ub=n, name="y")

    # Ordering constraints
    for i, j in G.edges():
        model.addConstr(y[i] + 1 <= y[j] + n * x[(i, j)], name=f"ord_{i}_{j}")

    # Objective: minimize number of backward edges
    model.setObjective(gp.quicksum(x[e] for e in G.edges()), GRB.MINIMIZE)

    start = time.perf_counter()
    model.optimize()
    runtime = time.perf_counter() - start

    obj_val = int(round(model.ObjVal)) if model.SolCount > 0 else None

    return obj_val, runtime


def iter_instances(root_dir: Path):
    """Yield (n, p, instance_id, file_path) for all edgelist files under root_dir."""
    for n_dir in sorted(root_dir.glob("n_*")):
        if not n_dir.is_dir():
            continue
        n = int(n_dir.name.split("_")[1])
        for p_dir in sorted(n_dir.glob("p_*/")):
            p = float(p_dir.name.split("_")[1])
            for file in sorted(p_dir.glob("instance_*.edgelist")):
                # Extract instance number
                inst_id = int(file.stem.split("_")[1])
                yield n, p, inst_id, file


def main():
    parser = argparse.ArgumentParser(
        description="Solve DFASP on generated Erdos–Rényi graph instances and create a CSV report."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "data"),
        help="Directory containing subfolders n_*/p_*/with edgelist instances.",
    )
    parser.add_argument(
        "--out-file",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "dfasp_results.csv"),
        help="Path to CSV file where results will be written.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress messages.",
    )

    args = parser.parse_args()

    root_dir = Path(args.data_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Data directory {root_dir} does not exist.")

    results = []
    
    # Get total number of instances for progress bar
    total_instances = sum(1 for _ in iter_instances(root_dir))
    
    # Create progress bar
    pbar = tqdm(iter_instances(root_dir), total=total_instances, 
                desc="Solving DFASP instances", unit="instance")

    for n, p, inst_id, file_path in pbar:
        # Update progress bar description with current instance info
        pbar.set_description(f"Solving n={n}, p={p}, instance={inst_id}")
        
        G = nx.read_edgelist(file_path, create_using=nx.DiGraph, nodetype=int)
        # Set per-instance time limit proportional to graph size: n/10 minutes
        time_limit_sec = (n / 10) * 30  # minutes -> seconds
        obj, runtime = solve_dfasp(G, silent=not args.verbose, time_limit=time_limit_sec)
        results.append((n, p, inst_id, obj, runtime))

        if args.verbose:
            print(f"obj={obj}, time={runtime:.2f}s")

    # Write CSV
    with open(args.out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "p", "instance", "objective", "runtime_sec"])
        writer.writerows(results)

    # Simple summary statistics
    summary = {}
    for n, p, *_ , obj, runtime in results:
        key = (n, p)
        if key not in summary:
            summary[key] = {"objs": [], "times": []}
        summary[key]["objs"].append(obj)
        summary[key]["times"].append(runtime)

    print("\n=== DFASP Experiment Summary ===")
    print(f"Total instances solved: {len(results)}")
    for (n, p), vals in sorted(summary.items()):
        avg_obj = sum(vals["objs"]) / len(vals["objs"])
        avg_time = sum(vals["times"]) / len(vals["times"])
        print(f"n={n:4d}, p={p:.1f} -> avg_obj={avg_obj:8.2f}, avg_time={avg_time:6.2f}s")

    print(f"\nDetailed results written to {args.out_file}")


if __name__ == "__main__":
    main() 