import numpy as np
import torch
from pathlib import Path
import re
from tqdm import tqdm
from Birkhoff_TSP import l2

def load_heatmaps(test_data_dir):
    """Load all heatmap files and sort by data index."""
    heatmap_files = list(Path(test_data_dir).glob('*-heatmap-*.npy'))
    print(f"Found {len(heatmap_files)} heatmap files in {test_data_dir}")
    
    # Debug: Show first few filenames
    if heatmap_files:
        print(f"First few files: {[f.name for f in heatmap_files[:5]]}")
    
    # Extract indices and sort files
    indexed_files = []
    for f in heatmap_files:
        match = re.search(r'-heatmap-(\d+)\.npy$', str(f))
        if match:
            idx = int(match.group(1))
            indexed_files.append((idx, f))
        else:
            print(f"Warning: Couldn't extract index from filename: {f.name}")
    
    indexed_files.sort(key=lambda x: x[0])
    print(f"Successfully parsed indices for {len(indexed_files)} files")
    
    heatmaps = []
    indices = []
    for idx, f in indexed_files:
        try:
            heatmap = np.load(str(f))
            print(f"Loaded {f.name} with shape {heatmap.shape}")
            
            if heatmap.shape == (1, 20, 20):  # Verify expected shape
                heatmaps.append(heatmap[0])  # Remove the singleton dimension
                indices.append(idx)
            elif heatmap.shape == (20, 20):  # Already in correct shape
                heatmaps.append(heatmap)
                indices.append(idx)
            else:
                print(f"Warning: Unexpected shape {heatmap.shape} for {f.name}, skipping")
        except Exception as e:
            print(f"Error loading {f.name}: {e}")
    
    return heatmaps, indices

def greedy_decode(heatmap, points):
    """
    Greedy decoding for TSP using heatmap scores.
    Args:
        heatmap: NxN array of edge scores
        points: Nx2 array of coordinates
    Returns:
        tour: List of indices representing the tour
    """
    N = len(points)
    
    # Debug: Check heatmap statistics
    if np.isnan(heatmap).any() or np.isinf(heatmap).any():
        print(f"Warning: Heatmap contains NaN or inf values")
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
                    scores[i,j] = (heatmap[i,j] + heatmap[j,i]) / dist
    
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

def compute_tour_length(tour, points):
    """Compute the total length of a tour."""
    length = 0.0
    for i in range(len(tour)):
        a = points[tour[i-1]]
        b = points[tour[i]]
        length += float(l2(torch.tensor(a), torch.tensor(b)))
    return length

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_dir', type=str, default='test_data', help='Directory containing heatmap files')
    parser.add_argument('--input_data', type=str, default='input_data/data_lst_20.pt', help='Path to original data points')
    parser.add_argument('--output', type=str, default='decoded_tours.pt', help='Output file for decoded tours')
    args = parser.parse_args()

    # Check if input data file exists
    if not Path(args.input_data).exists():
        print(f"Error: Input data file {args.input_data} does not exist!")
        return
    
    # Load original data points
    print(f"Loading data points from {args.input_data}...")
    data_points = torch.load(args.input_data)
    print(f"Loaded {len(data_points)} data points")
    
    # Check if test_data directory exists
    if not Path(args.test_data_dir).exists():
        print(f"Error: Test data directory {args.test_data_dir} does not exist!")
        return
    
    # Load heatmaps
    print("Loading heatmaps...")
    heatmaps, indices = load_heatmaps(args.test_data_dir)
    print(f"Loaded {len(heatmaps)} heatmaps with indices: {indices[:10]}...")  # Show first 10 indices
    
    if len(heatmaps) == 0:
        print("Error: No heatmaps found!")
        return
    
    # Decode tours and compute lengths
    decoded_tours = []
    tour_lengths = []
    
    print("Decoding tours...")
    for i, (heatmap, idx) in enumerate(tqdm(zip(heatmaps, indices))):
        try:
            if idx >= len(data_points):
                print(f"Warning: Heatmap index {idx} exceeds data points length {len(data_points)}, skipping...")
                continue
                
            points = data_points[idx].numpy()
            tour = greedy_decode(heatmap, points)
            length = compute_tour_length(tour, points)
            
            if np.isnan(length) or np.isinf(length):
                print(f"Warning: Invalid tour length {length} for index {idx}, skipping...")
                continue
                
            decoded_tours.append(tour)
            tour_lengths.append(length)
            
        except Exception as e:
            print(f"Error processing heatmap {idx}: {e}")
            continue
    
    if len(tour_lengths) == 0:
        print("Error: No valid tours were decoded!")
        return
        
    # Save results
    results = {
        'tours': decoded_tours,
        'lengths': tour_lengths,
        'indices': indices[:len(decoded_tours)]  # Only include successfully processed indices
    }
    torch.save(results, args.output)
    
    # Print statistics
    print(f"\nResults ({len(tour_lengths)} valid tours):")
    print(f"Average tour length: {np.mean(tour_lengths):.2f}")
    print(f"Min tour length: {np.min(tour_lengths):.2f}")
    print(f"Max tour length: {np.max(tour_lengths):.2f}")

if __name__ == '__main__':
    main() 