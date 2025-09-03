import numpy as np
import os
from datetime import datetime

def generate_random_qap(n, seed=None, value_range=(1, 100)):
    """Generate random QAP instance with uniform random matrices."""
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random distance matrix (symmetric)
    dist = np.random.randint(value_range[0], value_range[1], size=(n, n))
    dist = (dist + dist.T) // 2  # Make symmetric
    np.fill_diagonal(dist, 0)    # Zero diagonal
    
    # Generate random flow matrix (symmetric)
    flow = np.random.randint(value_range[0], value_range[1], size=(n, n))
    flow = (flow + flow.T) // 2  # Make symmetric
    np.fill_diagonal(flow, 0)    # Zero diagonal
    
    return dist, flow

def generate_structured_qap(n, seed=None, cluster_ratio=0.25):
    """Generate structured QAP instance with clustered patterns."""
    if seed is not None:
        np.random.seed(seed)
    
    # Generate distance matrix with clusters
    dist = np.zeros((n, n))
    cluster_size = max(1, int(n * cluster_ratio))
    
    # Calculate number of complete clusters and remaining elements
    num_complete_clusters = n // cluster_size
    remaining = n % cluster_size
    
    # Process complete clusters
    for i in range(num_complete_clusters):
        for j in range(num_complete_clusters):
            start_i = i * cluster_size
            start_j = j * cluster_size
            if i == j:
                # Within cluster: small distances
                dist[start_i:start_i+cluster_size, start_j:start_j+cluster_size] = \
                    np.random.randint(1, 20, size=(cluster_size, cluster_size))
            else:
                # Between clusters: larger distances
                dist[start_i:start_i+cluster_size, start_j:start_j+cluster_size] = \
                    np.random.randint(50, 100, size=(cluster_size, cluster_size))
    
    # Handle remaining elements if any
    if remaining > 0:
        last_cluster_start = num_complete_clusters * cluster_size
        # Fill remaining rows and columns
        for i in range(n):
            for j in range(n):
                if i >= last_cluster_start or j >= last_cluster_start:
                    if i == j:
                        dist[i,j] = np.random.randint(1, 20)
                    else:
                        dist[i,j] = np.random.randint(50, 100)
    
    # Make symmetric and zero diagonal
    dist = (dist + dist.T) // 2
    np.fill_diagonal(dist, 0)
    
    # Generate flow matrix with similar clustering pattern
    flow = np.zeros((n, n))
    
    # Process complete clusters
    for i in range(num_complete_clusters):
        for j in range(num_complete_clusters):
            start_i = i * cluster_size
            start_j = j * cluster_size
            if i == j:
                # Within cluster: high flows
                flow[start_i:start_i+cluster_size, start_j:start_j+cluster_size] = \
                    np.random.randint(50, 100, size=(cluster_size, cluster_size))
            else:
                # Between clusters: low flows
                flow[start_i:start_i+cluster_size, start_j:start_j+cluster_size] = \
                    np.random.randint(1, 20, size=(cluster_size, cluster_size))
    
    # Handle remaining elements if any
    if remaining > 0:
        last_cluster_start = num_complete_clusters * cluster_size
        # Fill remaining rows and columns
        for i in range(n):
            for j in range(n):
                if i >= last_cluster_start or j >= last_cluster_start:
                    if i == j:
                        flow[i,j] = np.random.randint(50, 100)
                    else:
                        flow[i,j] = np.random.randint(1, 20)
    
    # Make symmetric and zero diagonal
    flow = (flow + flow.T) // 2
    np.fill_diagonal(flow, 0)
    
    return dist, flow

def generate_sparse_qap(n, density=0.3, seed=None):
    """Generate sparse QAP instance with given density."""
    if seed is not None:
        np.random.seed(seed)
    
    # Generate sparse distance matrix
    dist = np.zeros((n, n))
    num_edges = int(n * n * density)
    edges = np.random.choice(n * n, num_edges, replace=False)
    for edge in edges:
        i, j = edge // n, edge % n
        if i != j:  # Don't add self-loops
            dist[i, j] = np.random.randint(1, 100)
    
    # Make symmetric
    dist = (dist + dist.T)
    
    # Generate sparse flow matrix
    flow = np.zeros((n, n))
    edges = np.random.choice(n * n, num_edges, replace=False)
    for edge in edges:
        i, j = edge // n, edge % n
        if i != j:  # Don't add self-loops
            flow[i, j] = np.random.randint(1, 100)
    
    # Make symmetric
    flow = (flow + flow.T)
    
    return dist, flow

def generate_scale_free_qap(n, seed=None):
    """Generate QAP instance with scale-free network characteristics."""
    if seed is not None:
        np.random.seed(seed)
    
    # Generate preferential attachment matrix
    dist = np.zeros((n, n))
    flow = np.zeros((n, n))
    
    # Start with a small clique
    initial_size = min(5, n)
    for i in range(initial_size):
        for j in range(i+1, initial_size):
            dist[i,j] = dist[j,i] = np.random.randint(1, 100)
            flow[i,j] = flow[j,i] = np.random.randint(1, 100)
    
    # Add remaining nodes with preferential attachment
    for i in range(initial_size, n):
        # Calculate attachment probabilities
        degrees = np.sum(dist > 0, axis=1)
        probs = degrees / np.sum(degrees)
        
        # Connect to m existing nodes
        m = min(3, i)
        targets = np.random.choice(i, m, p=probs[:i], replace=False)
        
        for j in targets:
            dist[i,j] = dist[j,i] = np.random.randint(1, 100)
            flow[i,j] = flow[j,i] = np.random.randint(1, 100)
    
    return dist, flow

def generate_geometric_qap(n, seed=None):
    """Generate QAP instance with geometric characteristics."""
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random points in 2D space
    points = np.random.rand(n, 2)
    
    # Calculate distance matrix based on Euclidean distances
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i,j] = int(np.linalg.norm(points[i] - points[j]) * 100)
    
    # Generate flow matrix with inverse relationship to distance
    flow = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                # Inverse relationship: closer points have higher flow
                flow[i,j] = int(100 / (dist[i,j] + 1))
    
    return dist, flow

def save_qap_instance(dist, flow, filename):
    """Save QAP instance to file in QAPLIB format."""
    n = len(dist)
    
    with open(filename, 'w') as f:
        # Write problem size
        f.write(f"{n}\n")
        
        # Write distance matrix
        for i in range(n):
            row = ' '.join(map(str, dist[i]))
            f.write(f"{row}\n")
        
        # Write flow matrix
        for i in range(n):
            row = ' '.join(map(str, flow[i]))
            f.write(f"{row}\n")

def main():
    # Create output directory
    output_dir = "input_data/synthetic"
    os.makedirs(output_dir, exist_ok=True)
    
    # Fixed size for all instances
    n = 1000
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Generating geometric QAP instances for facility location problems of size n={n}")
    print(f"Output directory: {output_dir}")
    print(f"Started at: {timestamp}")
    
    # Generate instances
    num_instances = 5
    
    instance_counter = 0
    while instance_counter < num_instances:
        # Random seed for each instance
        seed = instance_counter
        
        # Generate a geometric QAP instance for facility location
        dist, flow = generate_geometric_qap(n, seed=seed)
        
        # Save the instance with unique id (using 'geo' to indicate geometric type)
        filename = os.path.join(output_dir, f"qap_n{n}_geo_id{instance_counter}.dat")
        save_qap_instance(dist, flow, filename)
        
        if instance_counter % 1 == 0:  # Print progress for each instance since we only have 5
            print(f"Progress: {instance_counter + 1}/{num_instances} instances")
        
        instance_counter += 1
    
    end_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Generation complete! {num_instances} geometric QAP instances created.")
    print(f"Ended at: {end_timestamp}")

if __name__ == "__main__":
    main() 
