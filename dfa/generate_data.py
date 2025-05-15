import os
import argparse
import random
from typing import List, Optional

import networkx as nx


def generate_instances(ns: List[int], ps: List[float], num_instances: int, out_dir: str, seed: Optional[int] = None) -> None:
    """Generate Erdos–Rényi graph instances and write them as edgelist files.

    Parameters
    ----------
    ns : list[int]
        List of graph sizes (number of nodes).
    ps : list[float]
        List of edge creation probabilities.
    num_instances : int
        Number of instances to generate for each (n, p) pair.
    out_dir : str
        Directory where the instances will be written.
    seed : int | None, optional
        Global random seed. If provided, ensures reproducibility.
    """
    # Ensure deterministic behaviour for directory creation and other Python RNG users
    if seed is not None:
        random.seed(seed)

    for n in ns:
        for p in ps:
            # Construct output directory for this configuration
            cfg_dir = os.path.join(out_dir, f"n_{n}", f"p_{p}")
            os.makedirs(cfg_dir, exist_ok=True)

            for idx in range(num_instances):
                # Offset the seed per-instance for uniqueness yet reproducibility
                graph_seed = None if seed is None else seed + idx

                # Generate an undirected random graph first
                undirected_G = nx.erdos_renyi_graph(n=n, p=p, seed=graph_seed)

                # Randomly orient each edge to obtain a directed graph that can contain cycles
                G = nx.DiGraph()
                G.add_nodes_from(undirected_G.nodes())
                for u, v in undirected_G.edges():
                    # Flip a fair coin to decide orientation
                    if random.random() < 0.5:
                        G.add_edge(u, v)
                    else:
                        G.add_edge(v, u)

                file_path = os.path.join(cfg_dir, f"instance_{idx + 1}.edgelist")
                nx.write_edgelist(G, file_path, data=False)

                # Optionally store metadata for each instance (n, p, seed)
                meta_path = os.path.join(cfg_dir, f"instance_{idx + 1}.meta")
                with open(meta_path, "w", encoding="utf-8") as meta_file:
                    meta_file.write(f"n={n}\n")
                    meta_file.write(f"p={p}\n")
                    meta_file.write(f"seed={graph_seed}\n")

                print(f"Generated: {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Erdos–Rényi graph instances for DFA experiments."
    )
    parser.add_argument(
        "--ns",
        nargs="*",
        type=int,
        default=[20, 50, 100, 250],
        help="List of graph sizes (n values). Default: 20 50 100 250 500",
    )
    parser.add_argument(
        "--ps",
        nargs="*",
        type=float,
        default=[0.1, 0.5, 0.9],
        help="List of edge probabilities (p values). Default: 0.1 0.5 0.9",
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        default=20,
        help="Number of instances per (n, p). Default: 50",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "data"),
        help="Directory to store generated data. Default: <script_dir>/data",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. If omitted, uses non-deterministic RNG.",
    )

    args = parser.parse_args()

    generate_instances(args.ns, args.ps, args.num_instances, args.out_dir, args.seed) 