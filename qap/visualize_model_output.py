import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import random
from train_transformer_qap import SetTransformerQAP, parse_qaplib_file

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize output matrix from trained QAP model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the saved model checkpoint')
    parser.add_argument('--dataset', type=str, choices=['real', 'synthetic'], default='synthetic',
                        help='Dataset type to use for visualization')
    parser.add_argument('--device_idx', type=int, default=0,
                        help='Device index to use')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize')
    args = parser.parse_args()

    # Set up device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device_idx}")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU instead")

    # Set directory based on dataset type
    if args.dataset == 'real':
        qap_dir = "input_data/real/prob"
    else:  # synthetic
        qap_dir = "input_data/synthetic"

    # Load QAP instances
    all_files = os.listdir(qap_dir)
    dataset = []

    for fp in all_files:
        if fp.endswith('.dat'):
            n, A, B = parse_qaplib_file(os.path.join(qap_dir, fp))
            A = torch.tensor(A, dtype=torch.float32)
            B = torch.tensor(B, dtype=torch.float32)
            dataset.append((fp, n, A, B))

    # For synthetic data, split into train and validation
    if args.dataset == 'synthetic':
        # Shuffle dataset with fixed seed for reproducibility
        random.seed(42)
        random.shuffle(dataset)
        
        # Get validation set (same split as in training)
        split_idx = int(len(dataset) * 0.5)
        val_dataset = dataset[split_idx:]
        
        # Select random samples from validation set
        if len(val_dataset) > args.num_samples:
            samples = random.sample(val_dataset, args.num_samples)
        else:
            samples = val_dataset
    else:
        # For real data, use random samples
        if len(dataset) > args.num_samples:
            samples = random.sample(dataset, args.num_samples)
        else:
            samples = dataset

    # Load the model
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Extract model parameters from the checkpoint file name
    filename = os.path.basename(args.model_path)
    params = filename.split('_')
    
    # Find the d_model, n_heads, n_layers values
    d_model = 128  # Default
    n_heads = 4    # Default
    n_layers = 4   # Default
    
    # Try to extract from filename if possible
    try:
        # Typical format: best_model_2e-05_128_4_4_TIMESTAMP.pt
        lr_idx = next(i for i, param in enumerate(params) if 'e-' in param or '.' in param)
        if lr_idx + 3 < len(params):
            d_model = int(params[lr_idx + 1])
            n_heads = int(params[lr_idx + 2])
            n_layers = int(params[lr_idx + 3])
    except:
        print(f"Could not extract model parameters from filename. Using defaults: d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}")
    
    print(f"Model parameters: d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}")
    
    # Create model with the same architecture
    _, example_n, _, _ = samples[0]
    model = SetTransformerQAP(
        n=example_n,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        hidden_pair=d_model
    ).to(device)
    
    # Load state dict
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Create directory for saving visualizations
    os.makedirs("visualizations", exist_ok=True)
    
    # Visualize output matrix for each sample
    for i, (fp, n, A, B) in enumerate(samples):
        print(f"\nVisualizing sample {i+1}/{len(samples)}: {fp}")
        
        # Move data to device
        A = A.to(device)
        B = B.to(device)
        
        # Run inference
        with torch.no_grad():
            output_matrix = model(A.unsqueeze(0), B.unsqueeze(0)).squeeze(0)
        
        # Convert to numpy for visualization
        output_np = output_matrix.cpu().numpy()
        
        # Calculate sparsity (percentage of values close to zero)
        threshold = 0.01
        sparsity = np.sum(output_np < threshold) / (n * n) * 100
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Plot raw matrix
        plt.subplot(2, 2, 1)
        sns.heatmap(output_np, cmap='viridis')
        plt.title(f"Raw output matrix (N={n})\nSparsity: {sparsity:.2f}%")
        
        # Plot sorted values to check distribution
        plt.subplot(2, 2, 2)
        plt.plot(np.sort(output_np.flatten()))
        plt.title("Sorted matrix values")
        plt.grid(True)
        
        # Plot histogram of values
        plt.subplot(2, 2, 3)
        plt.hist(output_np.flatten(), bins=50)
        plt.title("Histogram of matrix values")
        plt.grid(True)
        
        # Plot distribution of row sums
        plt.subplot(2, 2, 4)
        row_sums = output_np.sum(axis=1)
        plt.bar(range(len(row_sums)), row_sums)
        plt.title("Row sums")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"visualizations/output_matrix_{i+1}_{fp.replace('.dat', '')}.png")
        plt.close()
        
        print(f"  Matrix shape: {output_np.shape}")
        print(f"  Sparsity: {sparsity:.2f}%")
        print(f"  Min value: {np.min(output_np):.6f}")
        print(f"  Max value: {np.max(output_np):.6f}")
        print(f"  Mean value: {np.mean(output_np):.6f}")
        print(f"  Visualization saved to visualizations/output_matrix_{i+1}_{fp.replace('.dat', '')}.png")

if __name__ == "__main__":
    main() 