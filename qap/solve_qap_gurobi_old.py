import gurobipy as gp
from gurobipy import GRB
import numpy as np
import os
import time
import logging
import signal
from datetime import datetime

# Set Gurobi license file location (adjust as needed)
os.environ['GRB_LICENSE_FILE'] = '/data/zhishang/be/gurobi.lic'

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
log_filename = f'qap_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}_gurobi_old.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)


class TimeoutError(Exception):
    """Custom exception raised when the hard time-out signal is triggered."""
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Process timed out")


# -----------------------------------------------------------------------------
# I/O utilities
# -----------------------------------------------------------------------------

def read_qap_instance(file_path):
    """Read a QAP instance in Burkard's *.dat format.

    Returns
    -------
    dist_matrix : (n, n) ndarray
        Inter-location distances.
    flow_matrix : (n, n) ndarray
        Inter-facility flows.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # strip comments / blanks
    lines = [ln.strip() for ln in lines if ln.strip() and not ln.startswith('#')]

    n = int(lines[0])
    dist_matrix = np.zeros((n, n))
    flow_matrix = np.zeros((n, n))

    # read square matrices row-wise
    idx = 1  # current line pointer

    # distance
    row, buf = 0, []
    while row < n:
        buf.extend(map(int, lines[idx].split()))
        if len(buf) >= n:
            dist_matrix[row] = buf[:n]
            buf = buf[n:]
            row += 1
        idx += 1

    # flow
    row, buf = 0, []
    while row < n:
        buf.extend(map(int, lines[idx].split()))
        if len(buf) >= n:
            flow_matrix[row] = buf[:n]
            buf = buf[n:]
            row += 1
        idx += 1

    return dist_matrix, flow_matrix


def read_optimal_solution(file_path):
    """Try to parse the optimum objective value from *.sln companion file."""
    try:
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
        return int(first_line.split()[1])  # second token
    except Exception as e:
        logging.warning(f"Could not read optimal solution from {file_path}: {e}")
        return None

# -----------------------------------------------------------------------------
# Solver routine (classic quadruple-sum formulation)
# -----------------------------------------------------------------------------

def solve_qap_instance(file_path, time_limit=None):
    """Solve the given QAP instance using the naïve MIQP objective.

    Parameters
    ----------
    file_path : str
        Path to the *.dat instance file.
    time_limit : int, optional
        Soft Gurobi time limit in seconds.  If omitted, we set it to 2n + 20.

    Returns
    -------
    dict
        Keys: success, objective_value, optimal_value, gap, runtime, solution, status / error.
    """
    try:
        # ------------------------------------------------------------------
        # Read instance and meta-data
        # ------------------------------------------------------------------
        dist, flow = read_qap_instance(file_path)
        n = len(dist)
        opt_file = file_path.replace('prob', 'sol').replace('.dat', '.sln')
        opt_val = read_optimal_solution(opt_file)

        # ------------------------------------------------------------------
        # Timing / timeout guards
        # ------------------------------------------------------------------
        start = time.time()
        if time_limit is None:
            time_limit = 2 * n + 20
        hard_limit = 6 * n  # seconds
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(hard_limit)

        # ------------------------------------------------------------------
        # Build model
        # ------------------------------------------------------------------
        model = gp.Model("QAP_old")
        x = model.addVars(n, n, vtype=GRB.BINARY, name="x")

        # assignment constraints (each facility to exactly 1 location and vice-versa)
        model.addConstrs((x.sum(i, '*') == 1 for i in range(n)), name="facility")
        model.addConstrs((x.sum('*', j) == 1 for j in range(n)), name="location")

        # ------------------------------------------------------------------
        # Classic quadratic objective  ∑_{i,k} ∑_{j,l} F_{ik} D_{jl} x_{ij} x_{kl}
        # ------------------------------------------------------------------
        quad_obj = gp.QuadExpr()
        for i in range(n):
            for k in range(n):
                # pre-fetch row for speed
                F_ik = flow[i, k]
                if F_ik == 0:
                    continue  # skip zeros early
                for j in range(n):
                    for l in range(n):
                        coeff = F_ik * dist[j, l]
                        if coeff == 0:
                            continue
                        quad_obj.addTerms([coeff], [x[i, j]], [x[k, l]])
        model.setObjective(quad_obj, GRB.MINIMIZE)

        model.setParam('TimeLimit', time_limit)
        logging.info(f"[old] Time limit set to {time_limit}s; hard limit {hard_limit}s; n={n}")

        # ------------------------------------------------------------------
        # Optimize
        # ------------------------------------------------------------------
        model.optimize()
        runtime = time.time() - start
        signal.alarm(0)  # disable alarm

        # ------------------------------------------------------------------
        # Return pack
        # ------------------------------------------------------------------
        if model.status == GRB.OPTIMAL:
            assignment = np.full(n, -1, dtype=int)
            for i in range(n):
                for j in range(n):
                    if x[i, j].X > 0.5:
                        assignment[i] = j
                        break
            gap = None
            if opt_val is not None:
                gap = (model.objVal - opt_val) / opt_val * 100
            return {
                'success': True,
                'objective_value': model.objVal,
                'optimal_value': opt_val,
                'gap': gap,
                'runtime': runtime,
                'solution': assignment,
                'status': 'Optimal'
            }
        else:
            return {
                'success': False,
                'error': f"Gurobi status: {model.status}",
                'runtime': runtime,
                'status': model.status
            }

    except TimeoutError:
        return {
            'success': False,
            'error': f"Process timed out after {hard_limit} seconds",
            'runtime': time.time() - start,
            'status': 'Timeout'
        }
    except Exception as ex:
        # make sure to disable alarm
        signal.alarm(0)
        return {'success': False, 'error': str(ex)}

# -----------------------------------------------------------------------------
# CLI driver – iterate through all instances in qap/input_data/prob
# -----------------------------------------------------------------------------

def main():
    prob_dir = 'qap/input_data/prob'
    instances = [f for f in os.listdir(prob_dir) if f.endswith('.dat')]
    instances.sort(key=lambda f: os.path.getsize(os.path.join(prob_dir, f)))

    for fname in instances:
        fpath = os.path.join(prob_dir, fname)
        logging.info(f"\n[old] Solving instance {fname}")
        res = solve_qap_instance(fpath)
        if res['success']:
            logging.info(f"Status: {res['status']}")
            logging.info(f"Objective value: {res['objective_value']}")
            if res['optimal_value'] is not None:
                logging.info(f"Optimal value: {res['optimal_value']}")
                logging.info(f"Gap: {res['gap']:.2f}%")
            else:
                logging.info("No optimal value available for gap calculation")
            logging.info(f"Runtime: {res['runtime']:.2f} s")
            logging.info(f"Solution: {res['solution']}")
        else:
            logging.error(f"Failed to solve {fname}: {res['error']}")


if __name__ == "__main__":
    main() 