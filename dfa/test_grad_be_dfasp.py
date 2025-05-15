import os
import sys
import time
import argparse
import datetime
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import scipy
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from IPF import ipf  # Re-use the IPF utility from QAP codebase


def compute_dfasp_cost_torch(A: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """Compute DFASP objective value – number of backward arcs.

    Parameters
    ----------
    A : (n, n) torch.Tensor
        Adjacency matrix of directed graph (0/1 or weighted).
    P : (n, n) torch.Tensor
        Permutation matrix.

    Returns
    -------
    torch.Tensor (scalar)
        Number (possibly weighted) of backward arcs.
    """
    # Ensure float32 for stable autograd behaviour
    A = A.to(dtype=torch.float32)
    P = P.to(dtype=torch.float32)

    # Permute adjacency — vertex order defined by permutation matrix
    A_perm = P.T @ A @ P  # shape (n, n)

    # Backward edges correspond to entries below main diagonal (row > col)
    backward = torch.tril(A_perm, diagonal=-1)
    return torch.sum(backward)


def parse_edgelist_to_tensor(filepath: str):
    """Load a NetworkX edgelist and return adjacency matrix as torch tensor.

    Returns
    -------
    n : int
    A : torch.Tensor of shape (n, n)
    """
    G = nx.read_edgelist(filepath, create_using=nx.DiGraph, nodetype=int)

    # Ensure deterministic node ordering (sorted by node label) for adjacency matrix
    nodes = sorted(G.nodes())
    n = len(nodes)
    A_np = nx.to_numpy_array(G, nodelist=nodes, dtype=np.float32)
    A = torch.tensor(A_np, dtype=torch.float32)
    return n, A


def setup_logging(lr, alg, device_idx):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("train_be_dfasp")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = log_dir / f"results_{lr}_dfasp_{alg}_{device_idx}_{timestamp}.log"
    return str(log_filename)


class MatrixModel(nn.Module):
    """Simple linear weight matrix constrained to Birkhoff polytope via custom projection."""

    def __init__(self, num_terms: int, alg: str):
        super().__init__()
        self.num_terms = num_terms
        self.mat = nn.Linear(num_terms, num_terms, bias=False)
        self.alg = alg

    def forward(self):
        # Project weights into the Birkhoff polytope depending on chosen algorithm.
        W = torch.abs(self.mat.weight)
        if self.alg == "gd":
            # Doubly-stochastic normalisation
            W = W / W.sum(dim=1, keepdim=True)
            W = W / W.sum(dim=0, keepdim=True)
        return W


def cont_Birkhoff_SFE_dfasp(W: torch.Tensor, k: int, A: torch.Tensor, perms: torch.Tensor,
                             setting, device="cpu"):
    """Continuous Birkhoff sampling with score-function estimator (adapted for DFASP)."""
    n = W.shape[0]
    fill = -n  # large negative value so unselected entries unlikely
    min_tl = float('inf')
    new_W = W.clone()
    cap = setting[1]
    min_P = None
    sum_thresh = 0
    total_loss = 0

    # Exploration noise
    perms = 0.80 * perms + 0.20 * torch.rand(n, n, device=device)

    # Pre-allocate permutation matrix
    P = torch.zeros(n, n, device=device)

    for idx in range(k):
        # Linear assignment on noisy weight matrix
        A_to_use = torch.where(new_W > 0, perms, fill).detach().cpu().numpy()
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(A_to_use, maximize=True)

        thresh = torch.amin(new_W[row_ind, col_ind])
        sum_thresh += thresh

        P.zero_()
        P[row_ind, col_ind] = 1

        new_W[row_ind, col_ind] -= thresh

        tl = compute_dfasp_cost_torch(A, P)

        if tl < min_tl:
            min_tl = tl
            min_P = P.clone()

        total_loss = total_loss + tl * thresh if idx > 0 else tl * thresh

    return min_tl, total_loss / sum_thresh, min_P, idx, sum_thresh.item()


def main():
    parser = argparse.ArgumentParser(description="Solve DFASP instances using gradient-based permutation learning algorithm.")
    parser.add_argument("--data-dir", type=str, default=os.path.join(os.path.dirname(__file__), "data"),
                        help="Directory containing graph instances (n_*/p_*/instance_*.edgelist).")
    parser.add_argument("--alg", type=str, required=True, help="Algorithm string, e.g. gd+k+10")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate for optimiser")
    parser.add_argument("--update_best_perms", type=int, default=1, help="Whether to update best permutations (1=yes)")
    parser.add_argument("--device_idx", type=int, required=True, help="Device index (integer)")

    args = parser.parse_args()

    alg = args.alg
    lr = args.lr
    device_idx = args.device_idx
    update_best_perms = args.update_best_perms == 1

    device = torch.device("cpu")
    T = 5000  # maximum gradient iterations

    # Gather dataset files
    root_dir = Path(args.data_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Data directory {root_dir} does not exist")

    instance_files = sorted(root_dir.glob("n_*/*/instance_*.edgelist"))
    if not instance_files:
        raise RuntimeError("No DFASP instance files found.")

    # Prepare logging
    log_filename = setup_logging(lr, alg, device_idx)
    with open(log_filename, 'w') as f:
        f.write(f"DFASP Training Log - {datetime.datetime.now()}\n")
        f.write(f"Parameters: lr={lr}, alg={alg}, device_idx={device_idx}\n\n")

    # Parse algorithm string (e.g., gd+k+10 or gd+p+0.2)
    alg_lst = alg.split("+")
    if len(alg_lst) < 3:
        raise ValueError("Algorithm string must follow pattern <base>+<k|p>+<value>")

    if alg_lst[1] == "k":
        setting = [alg_lst[1], int(alg_lst[2])]
    elif alg_lst[1] == "p":
        setting = [alg_lst[1], float(alg_lst[2])]
    else:
        raise ValueError("Unsupported setting specifier (use 'k' or 'p')")

    num_runs = 3  # repeat each instance multiple times

    for file_idx, file_path in enumerate(tqdm(instance_files, desc="Instances")):
        fp_str = str(file_path.relative_to(root_dir))
        print(f"\nProcessing {fp_str}")
        best_tl_overall = float('inf')
        best_run_idx = 0

        with open(log_filename, 'a') as f:
            f.write(f"\n{'='*50}\nProcessing {fp_str}\n{'='*50}\n")

        # Load adjacency matrix once for this instance
        n, A = parse_edgelist_to_tensor(file_path)
        A = A.to(device)

        for run_idx in range(num_runs):
            setting[1] = run_idx * 5 + 5  # vary hyper-parameter per run
            print(f"Run {run_idx + 1}/{num_runs} (setting={setting[1]})")

            num_terms = n
            run_start_time = time.time()
            # Time budget: n/10 minutes -> 30 seconds per 10 nodes = 3 * n seconds
            time_limit_sec = 3 * n

            # Initial permutation matrix approximation
            perms = ipf(torch.rand(num_terms, num_terms, device=device) / num_terms, 5, 1, 1)

            model = MatrixModel(num_terms, alg="gd").to(device)
            state_dict = model.state_dict()

            # Better weight initialisation via IPF
            weight = torch.rand(num_terms, num_terms, device=device)
            weight = torch.abs(weight)
            weight = weight / weight.sum(dim=1, keepdim=True)
            weight = weight / weight.sum(dim=0, keepdim=True)
            state_dict['mat.weight'] = ipf(weight, 10, 1, 1)
            model.load_state_dict(state_dict)

            # patience no longer used; rely on time budget
            best_tl = float('inf')
            best_perms = perms.clone()
            pt_en = 0
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

            for idx in range(T):
                W = model.forward()
                tl, loss, perms, num_P, sum_thresh = cont_Birkhoff_SFE_dfasp(
                    W, setting[1], A, best_perms, setting, device=device)
                loss.backward()

                # Check time budget
                if time.time() - run_start_time > time_limit_sec:
                    break  # stop when time budget exceeded

                if tl < best_tl:
                    best_tl = tl
                    if update_best_perms:
                        best_perms = perms
                    # re-initialise weights to encourage exploration
                    weight = torch.rand(num_terms, num_terms, device=device)
                    weight = torch.abs(weight)
                    weight = weight / weight.sum(dim=1, keepdim=True)
                    weight = weight / weight.sum(dim=0, keepdim=True)
                    state_dict['mat.weight'] = weight
                    model.load_state_dict(state_dict)
                    pt_en = 0
                    setting[1] = int(alg_lst[2])

                else:
                    loss_gap = (abs(loss - tl).item()) / (tl.item() + 0.001)
                    if loss_gap <= 0.005:
                        pt_en += 0.01
                    elif loss_gap >= 0.1:
                        pt_en -= 0.01

                    # dynamic adjustment of k / p parameter
                    if pt_en >= 1 and setting[1] <= num_terms:
                        setting[1] = int(setting[1] * 1.3) + 1
                        weight = torch.rand(num_terms, num_terms, device=device)
                        weight = torch.abs(weight)
                        weight = weight / weight.sum(dim=1, keepdim=True)
                        weight = weight / weight.sum(dim=0, keepdim=True)
                        state_dict['mat.weight'] = ipf(weight, 10, 1, 1)
                        model.load_state_dict(state_dict)
                        pt_en = 0

                    if pt_en <= -1 and setting[1] > int(alg_lst[2]):
                        setting[1] = int(max(setting[1] / 1.1, int(alg_lst[2])))
                        weight = torch.rand(num_terms, num_terms, device=device)
                        weight = torch.abs(weight)
                        weight = weight / weight.sum(dim=1, keepdim=True)
                        weight = weight / weight.sum(dim=0, keepdim=True)
                        state_dict['mat.weight'] = ipf(weight, 10, 1, 1)
                        model.load_state_dict(state_dict)
                        pt_en = 0

                # Optimiser step / PGD alternative
                if "pgd" in alg:
                    for param in model.parameters():
                        grad = param.grad.data
                        grad = grad - grad.mean(dim=1, keepdim=True)
                        grad = grad - grad.mean(dim=0, keepdim=True)
                        row, col = linear_sum_assignment(grad.cpu())
                        P = torch.zeros_like(W)
                        P[row, col] = 1
                        grad_norm = torch.norm(grad)
                        step_size = lr / (1 + 2 * grad_norm) if grad_norm > 0 else lr
                        if not hasattr(param, 'momentum_buffer'):
                            param.momentum_buffer = torch.zeros_like(param.data)
                        param.momentum_buffer = 0.9 * param.momentum_buffer + step_size * (P - param.data)
                        param.data = param.data + param.momentum_buffer
                        torch.nn.Module.zero_grad(model)
                else:
                    optimizer.step()
                    optimizer.zero_grad()

                if idx % 100 == 0 and idx > 0:
                    print(f"Iter {idx}: loss={loss.item():.4f}, tl={tl.item():.4f}, best={best_tl.item():.4f}, pt_en={pt_en:.2f}")

            # End of run; track best results
            if best_tl < best_tl_overall:
                best_tl_overall = best_tl
                best_run_idx = run_idx

        # Log summary for this instance
        with open(log_filename, 'a') as f:
            f.write(f"\nSummary for {fp_str}:\n")
            f.write(f"Best TL: {best_tl_overall}\n")
            f.write(f"Best run: {best_run_idx + 1}\n")
            f.write(f"{'='*50}\n")

    print(f"\nResults saved to {log_filename}")


if __name__ == "__main__":
    main() 