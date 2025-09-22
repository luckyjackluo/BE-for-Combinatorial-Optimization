import torch
import numpy as np
import os
import sys
import time
import argparse
import datetime
from tqdm import tqdm
from collections import OrderedDict
from scipy.optimize import linear_sum_assignment
from IPF import ipf

# Import from train_transformer_qap.py
from train_transformer_qap import SetTransformerQAP, parse_qaplib_file, compute_qap_cost_torch
from train_transformer_qap import cont_Birkhoff_SFE

def setup_logging(dataset_type, model_dir, device_idx):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"test_transformer/results_qap_transformer_{dataset_type}_{device_idx}_{timestamp}.log"
    os.makedirs("test_transformer", exist_ok=True)
    return log_filename

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test transformer model on QAP instances')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the saved model checkpoint directory')
    parser.add_argument('--checkpoint_type', type=str, choices=['best', 'last', 'epoch_100'], default='best',
                      help='Which checkpoint to load')
    parser.add_argument('--device_idx', type=int, default=0,
                      help='Device index to use')
    parser.add_argument('--d_model', type=int, default=128,
                      help='Dimension of the transformer model')
    parser.add_argument('--n_heads', type=int, default=4,
                      help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=4,
                      help='Number of transformer layers')
    parser.add_argument('--inference', type=str, choices=['zero', 'one'], default='zero',
                      help='If zero, just do inference. If one, continue training each instance for 1000 epochs')
    args = parser.parse_args()

    # Set up device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device_idx}")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU instead")
    
    # Set directory for the synthetic dataset
    qap_dir = "../input_data/qap/synthetic"

    # Setup logging
    log_filename = setup_logging("synthetic", args.model_path.split('/')[-1], args.device_idx)

    with open(log_filename, 'w') as f:
        f.write(f"Testing Log - {datetime.datetime.now()}\n")
        f.write(f"Model path: {args.model_path}, checkpoint type: {args.checkpoint_type}\n")
        f.write(f"Parameters: d_model={args.d_model}, n_heads={args.n_heads}, n_layers={args.n_layers}\n\n")

    # Load model checkpoint
    checkpoint_path = os.path.join(args.model_path, f"model_{args.checkpoint_type}.pt")
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint {checkpoint_path} not found. Checking for any .pt file in directory...")
        pt_files = [f for f in os.listdir(args.model_path) if f.endswith('.pt')]
        if pt_files:
            checkpoint_path = os.path.join(args.model_path, pt_files[0])
            print(f"Using checkpoint: {checkpoint_path}")
        else:
            raise FileNotFoundError(f"No checkpoint files found in {args.model_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Loaded checkpoint from {checkpoint_path}")

    # Load QAP instances
    all_files = os.listdir(qap_dir)
    dataset = []
    
    # Target specific instances
    target_ids = [352, 124, 384, 49, 158]
    
    for fp in all_files:
        if fp.endswith('.dat'):
            # Extract the ID from the filename
            try:
                file_id = int(fp.split('.d')[0].split('_')[-1].split('d')[-1])
                if file_id not in target_ids:
                    continue  # Skip this file if not in target_ids
            except (ValueError, IndexError):
                try:
                    file_id = int(fp.split('.d')[0].split('_')[-1].split('d')[-1])
                    if file_id not in target_ids:
                        continue
                except (ValueError, IndexError):
                    continue
            
            print(f"Found instance with ID {file_id}: {fp}")
            n, A, B = parse_qaplib_file(os.path.join(qap_dir, fp))
            A = torch.tensor(A, dtype=torch.float32)
            B = torch.tensor(B, dtype=torch.float32)
            dataset.append((fp, n, A, B, file_id))  # Add file_id for sorting
    
    # Sort by target_ids order to ensure consistent reporting
    dataset.sort(key=lambda x: target_ids.index(x[4]))
    
    # Create model with the same architecture as saved checkpoint
    example_n = dataset[0][1] if dataset else 0  # Get n from first instance
    model = SetTransformerQAP(
        n=example_n,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        hidden_pair=args.d_model
    ).to(device)
    
    # Load state dict
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Load best permutations if available in checkpoint
    val_best_perms = checkpoint.get('val_best_perms', {})
    val_best_tls = checkpoint.get('val_best_tls', {})
    
    # Set the k-value for cont_Birkhoff_SFE
    setting = ["k", 20]
    
    # Initialize best permutations for instances not in checkpoint
    for fp, n, _, _, _ in dataset:
        if fp not in val_best_perms:
            uniform_cpu = torch.ones(n, n) / n
            val_best_perms[fp] = ipf(uniform_cpu, num_it=5, row_sum=1, col_sum=1)
            val_best_tls[fp] = float('inf')
    
    print(f"\nTesting on {len(dataset)} instances\n")
    with open(log_filename, 'a') as f:
        f.write(f"Testing on {len(dataset)} instances\n\n")
    
    # Results storage
    results = []
    
    # Test each instance
    for idx, (fp, n, A, B, file_id) in enumerate(tqdm(dataset, desc="Testing")):
        print(f"\nTesting instance {idx+1}/{len(dataset)}: {fp} (ID: {file_id})")
        
        # Move data to device
        A = A.to(device)
        B = B.to(device)
        
        # Get best permutation for this instance
        perms = val_best_perms.get(fp, torch.ones(n, n) / n).to(device)
        
        # Time the inference
        start_time = time.time()
        
        if args.inference == 'zero':
            # Run model inference (same as before)
            with torch.no_grad():
                for trial in range(3):  # Run 3 trials, tracking the best result
                    # Forward pass through the model
                    W = model(A.unsqueeze(0), B.unsqueeze(0), perms.unsqueeze(0)).squeeze(0)
                    
                    # Apply Birkhoff algorithm
                    tl, loss, new_perm, _, sum_thresh = cont_Birkhoff_SFE(
                        W, setting[1], A, B, perms, setting, device.type
                    )
                    
                    # Update best permutation if improved
                    if tl < val_best_tls.get(fp, float('inf')):
                        val_best_tls[fp] = tl.item()
                        val_best_perms[fp] = new_perm.detach().cpu()
                        perms = new_perm.detach()
                    
                    print(f"  Trial {trial+1}: TL = {tl.item():.2f}, Loss = {loss.item():.4f}")
        else:  # inference == 'one'
            # Create a fresh copy of the model for this instance
            instance_model = SetTransformerQAP(
                n=n,
                d_model=args.d_model,
                n_heads=args.n_heads,
                n_layers=args.n_layers,
                hidden_pair=args.d_model
            ).to(device)
            
            # Load pre-trained weights
            instance_model.load_state_dict(model.state_dict())
            
            # Create optimizer for this instance
            optimizer = torch.optim.Adam(instance_model.parameters(), lr=1e-3)
            
            # Track best results
            best_tl = val_best_tls.get(fp, float('inf'))
            best_perm = perms.clone()
            
            # Train for 1000 epochs
            print(f"  Training on instance {fp} for 1000 epochs")
            pbar = tqdm(range(500), desc="Training")
            for epoch in pbar:
                # Forward pass
                instance_model.train()
                W = instance_model(A.unsqueeze(0), B.unsqueeze(0), perms.unsqueeze(0)).squeeze(0)
                
                # Apply Birkhoff algorithm
                tl, loss, new_perm, _, _ = cont_Birkhoff_SFE(
                    W, setting[1], A, B, perms, setting, device.type
                )
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update best permutation if improved
                # if tl < best_tl:
                #     best_tl = tl.item()
                #     best_perm = new_perm.detach()
                #     perms = best_perm.clone()
                
                # Update progress bar
                if epoch % 10 == 0:
                    pbar.set_postfix({
                        'loss': loss.item(),
                        'tl': tl.item(),
                        'best_tl': best_tl
                    })
            
            # Update the best results
            val_best_tls[fp] = best_tl
            val_best_perms[fp] = best_perm.cpu()
            
            print(f"  Training completed. Best TL = {best_tl:.4f}")
        
        inference_time = time.time() - start_time
        
        # Store results
        result = {
            'filename': fp,
            'id': file_id,
            'n': n,
            'tl': val_best_tls[fp],
            'time': inference_time
        }
        results.append(result)
        
        # Log results
        with open(log_filename, 'a') as f:
            f.write(f"Instance {fp} (ID: {file_id}):\n")
            f.write(f"  Size: {n}x{n}\n")
            f.write(f"  Best TL: {val_best_tls[fp]:.4f}\n")
            f.write(f"  Inference time: {inference_time:.2f}s\n\n")
        
        print(f"  Best TL: {val_best_tls[fp]:.4f}")
        print(f"  Inference time: {inference_time:.2f}s")
    
    # Print summary
    print("\nSummary:")
    total_tl = sum(r['tl'] for r in results)
    avg_tl = total_tl / len(results) if results else 0
    
    with open(log_filename, 'a') as f:
        f.write("\nSummary:\n")
        f.write("=========\n")
        f.write(f"{'ID':<8}{'Size':<8}{'Best TL':<15}{'Time (s)':<10}\n")
        
        for r in results:
            f.write(f"{r['id']:<8}{r['n']:<8}{r['tl']:<15.4f}{r['time']:<10.2f}\n")
        
        f.write(f"\nAverage TL: {avg_tl:.4f}\n")
    
    print(f"Average TL: {avg_tl:.4f}")
    print(f"Results saved to {log_filename}")

if __name__ == "__main__":
    main() 