import numpy as np
import torch
from tqdm import tqdm
import sys
from pathlib import Path

# Import distance function from Birkhoff_TSP
sys.path.append(str(Path(__file__).parent))
from Birkhoff_TSP import get_l2_dist

def parse_concorde_line(line):
    line = line.strip()
    points_str = line.split(' output ')[0]
    points = points_str.split(' ')
    points = np.array([[float(points[i]), float(points[i + 1])] for i in range(0, len(points), 2)])
    tour_str = line.split(' output ')[1]
    tour = tour_str.split(' ')
    tour = np.array([int(t) for t in tour if t.strip() != ''])
    tour -= 1  # zero-based
    return points, tour

def tour_length(points, tour):
    # Compute the total length of the tour
    length = 0.0
    for i in range(len(tour)):
        a = points[tour[i-1]]
        b = points[tour[i]]
        length += np.linalg.norm(a - b)
    return length

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to Concorde-style TSP data file')
    parser.add_argument('--output_dir', type=str, default='input_data', help='Directory to save torch files')
    parser.add_argument('--num_terms', type=int, required=True, help='Number of points per instance')
    parser.add_argument('--max_instances', type=int, default=500, help='How many instances to process (default: 500)')
    args = parser.parse_args()

    # Read all lines
    with open(args.input, 'r') as f:
        lines = [l for l in f if l.strip()]

    data_lst = []
    cost_lst = []
    dist_lst = []
    tour_lst = []
    input_lst = []

    for idx, line in enumerate(tqdm(lines[:args.max_instances])):
        points, tour = parse_concorde_line(line)
        if len(points) != args.num_terms:
            continue  # skip malformed
        points_tensor = torch.tensor(points, dtype=torch.float32)
        tour_tensor = torch.tensor(tour, dtype=torch.long)
        D = get_l2_dist(points_tensor)
        cost = tour_length(points, tour)
        data_lst.append(points_tensor)
        cost_lst.append(cost)
        dist_lst.append(D)
        tour_lst.append(tour_tensor)
        input_lst.append((points_tensor, cost, D))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    torch.save(data_lst, output_dir / f'data_lst_{args.num_terms}.pt')
    torch.save(cost_lst, output_dir / f'cost_lst_{args.num_terms}.pt')
    torch.save(dist_lst, output_dir / f'dist_lst_{args.num_terms}.pt')
    torch.save(tour_lst, output_dir / f'tour_lst_{args.num_terms}.pt')
    torch.save(input_lst, output_dir / f'input_lst_{args.num_terms}.pt')
    print(f"Saved {len(data_lst)} instances to {output_dir}")

if __name__ == '__main__':
    main() 