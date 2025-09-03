import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
import sys
import time
import argparse
import datetime
import copy
from scipy.optimize import linear_sum_assignment
from collections import OrderedDict
from IPF import ipf  # For iterative proportional fitting to obtain a doubly-stochastic matrix
import random
from tqdm import tqdm

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2500):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        output = torch.matmul(attention, V)
        
        return output, attention
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # Linear projections
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attended, attention = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Output projection
        output = self.W_o(attended)
        
        return output, attention

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=2048):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
    
    def forward(self, x, mask=None):
        # Self attention
        attended, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attended))
        
        # Feed forward
        fed_forward = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(fed_forward))
        
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, n_layers):
        super().__init__()
        
        # Use deep copies of the provided encoder_layer to ensure each layer has independent parameters
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(n_layers)])
        
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class MatrixToCrossAttentionFusion(nn.Module):
    def __init__(self, n_range=(20, 50), d_model=512, n_heads=8, n_layers=6):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_n = n_range[1]
        
        # Token embeddings
        self.embed1 = nn.Linear(1, d_model)
        self.embed2 = nn.Linear(1, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.max_n**2)
        
        # Encoder layers
        encoder_layer1 = TransformerEncoderLayer(d_model, n_heads)
        encoder_layer2 = TransformerEncoderLayer(d_model, n_heads)
        
        self.encoder1 = TransformerEncoder(encoder_layer1, n_layers)
        self.encoder2 = TransformerEncoder(encoder_layer2, n_layers)
        
        # Cross attention
        self.cross_attention = MultiHeadAttention(d_model, n_heads)
        
        # Fusion layer
        fusion_layer = TransformerEncoderLayer(d_model, n_heads)
        self.fusion = TransformerEncoder(fusion_layer, 2)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, 1)
        
    def forward(self, matrix1, matrix2):
        batch_size, n, _ = matrix1.shape
        
        # Flatten and embed
        tokens1 = matrix1.reshape(batch_size, n*n, 1)
        tokens2 = matrix2.reshape(batch_size, n*n, 1)
        
        # Token embedding
        emb1 = self.embed1(tokens1)  # [batch, n*n, d_model]
        emb2 = self.embed2(tokens2)  # [batch, n*n, d_model]
        
        # Add positional encoding
        x1 = self.pos_encoder(emb1)
        x2 = self.pos_encoder(emb2)
        
        # Encode separately
        enc1 = self.encoder1(x1)  # [batch, n*n, d_model]
        enc2 = self.encoder2(x2)  # [batch, n*n, d_model]
        
        # Cross attention
        cross_attended, _ = self.cross_attention(enc2, enc1, enc1)
        
        # Combine with residual
        combined = cross_attended + enc2
        
        # Fusion
        fused = self.fusion(combined)
        
        # Project to output
        output_tokens = self.output_proj(fused)
        
        # Reshape to matrix
        output_matrix = output_tokens.reshape(batch_size, n, n)
        
        return output_matrix

def compute_qap_cost_torch(A, B, P):
    """Compute QAP cost using torch tensors."""
    A = A.to(dtype=torch.float32)
    B = B.to(dtype=torch.float32)
    P = P.to(dtype=torch.float32)
    PBPT = P @ B @ P.T
    return torch.sum(A * PBPT)

def parse_qaplib_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    numbers = list(map(float, content.split()))
    n = int(numbers[0])
    A_flat = numbers[1:n*n+1]
    B_flat = numbers[n*n+1:]
    
    A = np.array(A_flat).reshape((n, n))
    B = np.array(B_flat).reshape((n, n))
    
    return n, A, B

def parse_sln_file(filepath):
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
    optimal_value = int(first_line.split()[1])
    return optimal_value

def setup_logging(dataset_type, lr, device_idx):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"train_transformer/results_{lr}_qap_transformer_{dataset_type}_{device_idx}_{timestamp}.log"
    os.makedirs("train_transformer", exist_ok=True)
    return log_filename

def compute_validation_loss(model, val_dataset, device, setting, val_best_perms, val_best_tls):
    """Compute average loss on validation set and update best permutations when better ones are found.
    """
    val_loss = 0.0
    val_tl = 0.0
    
    for fp, n, A, B in val_dataset:
        # Move matrices to device
        A = A.to(device, non_blocking=True)
        B = B.to(device, non_blocking=True)
            
        # Fetch the current best permutation for this instance
        perms = val_best_perms[fp].to(device, non_blocking=True)

        # Forward pass - pass perms as additional input
        W = model(A.unsqueeze(0), B.unsqueeze(0), perms.unsqueeze(0)).squeeze(0)
            
        # Compute QAP cost and loss, potentially updating best permutation
        tl, loss, new_perm, _, _ = cont_Birkhoff_SFE(
            W, setting[1], A, B, perms, setting, device.type
            )
            
        val_loss += loss.item()
        val_tl += tl.item()
    
        # Update stored best permutation if improvement is found
        if tl < val_best_tls[fp] and new_perm is not None:
            val_best_tls[fp] = tl.item()
            val_best_perms[fp] = new_perm.detach().cpu()

    return val_loss / len(val_dataset), val_tl / len(val_dataset)

def two_approx_max_weight_perfect_matching(W, B_allowed):
    """
    2-approx MWPM for a bipartite graph G.
    W: (n x n) float Tensor on GPU (weights).
    B_allowed: (n x n) bool Tensor on GPU (allowed edges).

    Returns:
        matching_indices: list of (col, row) pairs
        total_weight: sum of chosen edges' weights
    """
    n = W.shape[0]
    row_used = torch.zeros(n, dtype=torch.bool, device=W.device)
    col_used = torch.zeros(n, dtype=torch.bool, device=W.device)

    matching_indices = []
    total_weight = 0.0

    for c in range(n):
        if col_used[c]:
            continue
        
        valid_mask = (~row_used) & B_allowed[:, c]
        if not valid_mask.any():
            continue
        
        # Among valid rows, pick the one with max weight
        valid_weights = W[:, c].clone()
        valid_weights[~valid_mask] = float('-inf')
        
        i_star = torch.argmax(valid_weights).item()
        total_weight += W[i_star, c].item()
        matching_indices.append((c, i_star))
        
        row_used[i_star] = True
        col_used[c] = True

    return matching_indices, total_weight

def cont_Birkhoff_SFE(
        W, k, A, B, perms, setting,
        device="cuda"  # This is a device_type string, not a torch.device
):
    """
    Continuous Birkhoff SFE using the *original* 2-approx MWPM
    instead of SciPy's Hungarian solver.

    Parameters
    ----------
    W        : (n,n) tensor, mutable copy will be made
    k        : int         – number of successive matchings
    A, B     : matrices for QAP cost
    perms    : (n,n) tensor – base permutation weights
    setting  : tuple/list   – as before (cap etc.)
    device   : "cuda" | "cpu"
    """
    n = W.shape[0]
    fill = -n                       # sentinel for disallowed edge
    min_tl   = float("inf")
    new_W    = W.clone()
    cap      = setting[1]           # kept for interface parity
    min_P    = None
    sum_thresh = 0.0
    total_loss = 0.0

    # add exploration noise
    #perms = 0.90 * perms + 0.10 * torch.rand(n, n, device=perms.device)

    P = torch.zeros(n, n, device=W.device)  # reusable permutation matrix

    for idx in range(k):
        # Check if the graph is feasible for a perfect matching
        B_allowed = (new_W > 0)
        rows_with_edges = torch.any(B_allowed, dim=1)
        cols_with_edges = torch.any(B_allowed, dim=0)
        
        # If any row or column has no valid edges, early stop
        if not (torch.all(rows_with_edges) and torch.all(cols_with_edges)):
            # Early stop - graph is not feasible for a perfect matching
            break
        
        # --------------------------------------------------------------
        #   2-approx maximum-weight perfect matching (column version)
        #   weights = perms, allowed mask = new_W > 0
        # --------------------------------------------------------------
        matching, _ = two_approx_max_weight_perfect_matching(perms, B_allowed)

        if len(matching) < n:  # should never occur if graph is feasible
            # Instead of error, early stop the loop
            break

        cols = torch.tensor([c for c, r in matching],
                            device=W.device, dtype=torch.long)
        rows = torch.tensor([r for c, r in matching],
                            device=W.device, dtype=torch.long)

        # threshold
        thresh = torch.amin(new_W[rows, cols])
        sum_thresh += thresh

        # build permutation matrix
        P.zero_()
        P[rows, cols] = 1.0

        # update residual weights
        new_W[rows, cols] -= thresh

        # QAP cost
        tl = compute_qap_cost_torch(A, B, P)

        if tl < min_tl:
            min_tl = tl
            min_P  = P.clone()

        total_loss = total_loss + tl * thresh if idx else tl * thresh

    # Make sure we have valid outputs even if we stopped early
    if sum_thresh == 0.0:
        # We didn't complete any iteration, so create a default output
        # Use linear_sum_assignment as fallback
        A_to_use = perms.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(A_to_use, maximize=True)
        
        P.zero_()
        P[row_ind, col_ind] = 1.0
        
        tl = compute_qap_cost_torch(A, B, P)
        min_tl = tl
        min_P = P.clone()
        total_loss = tl
        sum_thresh = 1.0

    # Return the best results we found (even if we stopped early)
    return min_tl, total_loss / max(sum_thresh, 1e-8), min_P, idx, sum_thresh.item()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Solve QAP instances using transformer model')
    parser.add_argument('--dataset', type=str, choices=['real', 'synthetic'], default='real',
                      help='Dataset type to solve (real or synthetic)')
    parser.add_argument('--lr', type=float, required=True,
                      help='Learning rate')
    parser.add_argument('--device_idx', type=int, required=True,
                      help='Device index to use')
    parser.add_argument('--d_model', type=int, default=512,
                      help='Dimension of the transformer model')
    parser.add_argument('--n_heads', type=int, default=8,
                      help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=6,
                      help='Number of transformer layers')
    parser.add_argument('--load_model', type=str, default=None,
                      help='Path to a saved model checkpoint directory to resume training from')
    parser.add_argument('--checkpoint_type', type=str, choices=['best', 'last'], default=None,
                      help='Which checkpoint to load when resuming training ("best" or "last")')
    parser.add_argument('--evaluate_only', action='store_true',
                      help='Only evaluate the model on validation data, no training')
    args = parser.parse_args()

    # Validate arguments
    if args.checkpoint_type and not args.load_model:
        raise ValueError("--checkpoint_type can only be used with --load_model")

    # Set up device for GPU training
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device_idx}")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU instead")
    
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Set directory based on dataset type
    if args.dataset == 'real':
        qap_dir = "input_data/real/prob"
        sol_dir = "input_data/real/sol"
    else:  # synthetic
        qap_dir = "input_data/synthetic"
        sol_dir = None

    # Load QAP instances
    all_files = os.listdir(qap_dir)
    dataset = []
    optimal_values = {}

    for fp in all_files:
        if fp.endswith('.dat'):
            n, A, B = parse_qaplib_file(os.path.join(qap_dir, fp))
            A = torch.tensor(A, dtype=torch.float32)
            B = torch.tensor(B, dtype=torch.float32)
            dataset.append((fp, n, A, B))
            
            if args.dataset == 'real':
                sol_file = fp.replace('.dat', '.sln')
                if os.path.exists(os.path.join(sol_dir, sol_file)):
                    optimal_values[fp] = parse_sln_file(os.path.join(sol_dir, sol_file))

    # For synthetic data, split into train and validation
    if args.dataset == 'synthetic':
        # Shuffle dataset with fixed seed for reproducibility
        random.seed(42)
        random.shuffle(dataset)
        dataset = dataset[:200]
        # Split dataset: 50% training, 50% validation
        split_idx = int(len(dataset) * 0.5)
        train_dataset = dataset[:split_idx]
        val_dataset = dataset[split_idx:]
        
        print(f"Dataset split: {len(train_dataset)} training instances, {len(val_dataset)} validation instances")
    else:
        # For real data, use all instances for training
        train_dataset = dataset
        val_dataset = []

    # Setup logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = setup_logging(args.dataset, args.lr, args.device_idx)
    
    # Create model directory for saving best and last checkpoints
    model_dir = f"train_transformer/model_{args.lr}_{args.d_model}_{args.n_heads}_{args.n_layers}_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)

    with open(log_filename, 'w') as f:
        f.write(f"Training Log - {datetime.datetime.now()}\n")
        f.write(f"Parameters: dataset={args.dataset}, lr={args.lr}, d_model={args.d_model}, "
                f"n_heads={args.n_heads}, n_layers={args.n_layers}\n")
        if args.load_model:
            f.write(f"Resuming training from model: {args.load_model}\n")
            if args.checkpoint_type:
                f.write(f"Using checkpoint type: {args.checkpoint_type}\n")
        if args.evaluate_only:
            f.write(f"Evaluation mode only, no training\n")
        if args.dataset == 'synthetic':
            f.write(f"Dataset: {len(train_dataset)} training instances, {len(val_dataset)} validation instances\n\n")

    # Initialize results storage
    # In synthetic mode, we'll track the best model across all instances
    best_val_loss = float('inf')
    best_model_state = None
    start_epoch = 0
    
    # Create a single model for all instances
    # Get the dimensionality from the first instance
    _, example_n, _, _ = train_dataset[0]
    
    # Initialize Set-Transformer based model
    model = SetTransformerQAP(
        n=example_n,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        hidden_pair=args.d_model  # use model dimension as hidden size for MLP
    ).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Load saved model if provided
    if args.load_model and os.path.exists(args.load_model):
        print(f"Loading model from {args.load_model}")
        
        # Determine which checkpoint to load
        checkpoint_type = args.checkpoint_type or 'best'  # Default to 'best' if not specified
        checkpoint_path = os.path.join(args.load_model, f"model_{checkpoint_type}.pt")
        
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint {checkpoint_path} not found. Checking for any .pt file in directory...")
            pt_files = [f for f in os.listdir(args.load_model) if f.endswith('.pt')]
            if pt_files:
                checkpoint_path = os.path.join(args.load_model, pt_files[0])
                print(f"Using checkpoint: {checkpoint_path}")
            else:
                raise FileNotFoundError(f"No checkpoint files found in {args.load_model}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
        best_val_loss = checkpoint['val_loss'] if 'val_loss' in checkpoint else float('inf')
        print(f"Resuming from epoch {start_epoch}, best validation loss: {best_val_loss:.4f}")
        
        # Log the loaded model information
        with open(log_filename, 'a') as f:
            f.write(f"Loaded model from {checkpoint_path}\n")
            f.write(f"Resuming from epoch {start_epoch}, best validation loss: {best_val_loss:.4f}\n")
    
    # Evaluation only mode
    if args.evaluate_only:
        if not args.load_model:
            raise ValueError("--evaluate_only requires --load_model to be specified")
        
        if val_dataset:
            print("\nEvaluating model on validation data...")
            # Set the k-value for cont_Birkhoff_SFE
            setting = ["k", 10]
            
            val_loss, val_tl = compute_validation_loss(model, val_dataset, device, setting, val_best_perms, val_best_tls)
            
            print(f"Validation loss: {val_loss:.4f}, Validation TL: {val_tl:.4f}")
            with open(log_filename, 'a') as f:
                f.write(f"\nEvaluation results: Validation loss: {val_loss:.4f}, Validation TL: {val_tl:.4f}\n")
        else:
            print("No validation data available for evaluation")
        
        return
    
    # Training parameters
    max_epochs = 1000
    patience = 500  # Number of epochs with no improvement after which training stops
    epochs_no_improve = 0
    
    # Store performance history
    train_losses = []
    val_losses = []
    
    # Training loop
    print(f"\nStarting training with {len(train_dataset)} instances from epoch {start_epoch+1}")
    train_start_time = time.time()

    # ---------------------------------------------------------------------------------
    # Initialise per-instance best permutations (uniform doubly-stochastic matrices)
    # ---------------------------------------------------------------------------------
    best_perms = {}
    best_tls = {}
    for fp_i, n_i, _, _ in train_dataset:
        uniform_cpu = torch.ones(n_i, n_i) / n_i
        best_perms[fp_i] = ipf(uniform_cpu, num_it=5, row_sum=1, col_sum=1)
        best_tls[fp_i] = float('inf')
    
    # Also initialize best permutations for validation instances
    val_best_perms = {}
    val_best_tls = {}
    for fp_i, n_i, _, _ in val_dataset:
        uniform_cpu = torch.ones(n_i, n_i) / n_i
        val_best_perms[fp_i] = ipf(uniform_cpu, num_it=5, row_sum=1, col_sum=1)
        val_best_tls[fp_i] = float('inf')
    
    for epoch in range(start_epoch, max_epochs):
        print(f"\nEpoch {epoch+1}/{max_epochs}")
        epoch_loss = 0.0
        epoch_tl = 0.0
        epoch_start_time = time.time()
        
        # Shuffle training data for this epoch
        random.shuffle(train_dataset)
        
        # Process each instance in the training set with tqdm
        batch_loss = 0.0
        batch_tl = 0.0
        batch_count = 0
        
        for batch_idx, (fp, n, A, B) in enumerate(tqdm(train_dataset, desc="Training")):
            # Run single iteration for this problem
            instance_start_time = time.time()
            
            # Move data to device
            A = A.to(device, non_blocking=True)
            B = B.to(device, non_blocking=True)
            
            # Fetch the current best permutation for this instance
            perms = best_perms[fp].to(device, non_blocking=True)
            
            # Set the k-value for cont_Birkhoff_SFE
            setting = ["k", 20]
            
            # Forward pass through the model - pass perms as additional input
            W = model(A.unsqueeze(0), B.unsqueeze(0), perms.unsqueeze(0)).squeeze(0)
            
            # Compute QAP cost and loss, and potentially update best_perms
            tl, loss, new_perm, _, _ = cont_Birkhoff_SFE(
                W, setting[1], A, B, perms, setting, device.type
            )
            
            # Accumulate loss and metrics
            batch_loss += loss
            batch_tl += tl
            batch_count += 1
            
            # Every 10 instances, perform backward pass and optimization
            if batch_count == 10 or batch_idx == len(train_dataset) - 1:
                # Average the accumulated loss
                avg_batch_loss = batch_loss / batch_count
                
                # Backward pass
                optimizer.zero_grad()
                avg_batch_loss.backward()
                optimizer.step()
                
                # Reset batch accumulators
                batch_loss = 0.0
                batch_tl = 0.0
                batch_count = 0
            
            # Accumulate epoch metrics
            epoch_loss += loss.item()
            epoch_tl += tl
            
            # Update stored best permutation if improvement is found
            if tl < best_tls[fp] and new_perm is not None:
                best_tls[fp] = tl.item()
                best_perms[fp] = new_perm.detach().cpu()
        
        # Compute average metrics for the epoch
        avg_train_loss = epoch_loss / len(train_dataset)
        avg_train_tl = epoch_tl / len(train_dataset)
        train_losses.append(avg_train_loss)
        epoch_time = time.time() - epoch_start_time
        
        # Log epoch results
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s - Train loss: {avg_train_loss:.4f}, Train TL: {avg_train_tl:.4f}")
        
        with open(log_filename, 'a') as f:
            f.write(f"\nEpoch {epoch+1} - Time: {epoch_time:.2f}s, Train loss: {avg_train_loss:.4f}, Train TL: {avg_train_tl:.4f}\n")
        
        # Validation phase
        if val_dataset:
            val_start_time = time.time()
            val_loss, val_tl = compute_validation_loss(model, val_dataset, device, setting, val_best_perms, val_best_tls)
            val_losses.append(val_loss)
            val_time = time.time() - val_start_time
            
            print(f"Validation completed in {val_time:.2f}s - Validation loss: {val_loss:.4f}, Validation TL: {val_tl:.4f}")
            
            with open(log_filename, 'a') as f:
                f.write(f"Epoch {epoch+1} - Validation loss: {val_loss:.4f}, Validation TL: {val_tl:.4f}\n")
            
            # Save best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'train_loss': avg_train_loss,
                    'val_loss': val_loss,
                    'train_tl': avg_train_tl,
                    'val_tl': val_tl,
                    'train_best_perms': best_perms,
                    'train_best_tls': best_tls,
                    'val_best_perms': val_best_perms,
                    'val_best_tls': val_best_tls
                }
                
                # Save best model checkpoint
                best_model_path = os.path.join(model_dir, "model_best.pt")
                torch.save(best_model_state, best_model_path)
                print(f"New best validation loss: {best_val_loss:.4f} - Saved model to {best_model_path}")
                
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve} epochs")
            
            # Early stopping
            if epochs_no_improve >= patience:
                print(f"Early stopping after {epoch+1} epochs without improvement")
                break

        # # Save last checkpoint every 10 epochs
        # if (epoch + 1) % 10 == 0:
        #     last_model_path = os.path.join(model_dir, "model_last.pt")
        #     torch.save({
        #         'model': model.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'epoch': epoch,
        #         'train_loss': avg_train_loss,
        #         'val_loss': val_loss if val_dataset else None,
        #         'train_tl': avg_train_tl, 
        #         'val_tl': val_tl if val_dataset else None,
        #         'train_best_perms': best_perms,
        #         'train_best_tls': best_tls,
        #         'val_best_perms': val_best_perms,
        #         'val_best_tls': val_best_tls
        #     }, last_model_path)
        #     print(f"Saved last checkpoint at epoch {epoch+1} to {last_model_path}")
        
        # Save numbered checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            epoch_model_path = os.path.join(model_dir, f"model_epoch_{epoch+1}.pt")
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': val_loss if val_dataset else None,
                'train_tl': avg_train_tl, 
                'val_tl': val_tl if val_dataset else None,
                'train_best_perms': best_perms,
                'train_best_tls': best_tls,
                'val_best_perms': val_best_perms,
                'val_best_tls': val_best_tls
            }, epoch_model_path)
            print(f"Saved epoch {epoch+1} checkpoint to {epoch_model_path}")

    # Save the final (last) model
    if args.dataset == 'synthetic':
        # Save last model state
        last_model_path = os.path.join(model_dir, "model_last.pt")
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': val_loss if val_dataset else None,
            'train_tl': avg_train_tl,
            'val_tl': val_tl if val_dataset else None,
            'train_best_perms': best_perms,
            'train_best_tls': best_tls,
            'val_best_perms': val_best_perms,
            'val_best_tls': val_best_tls
        }, last_model_path)
        
        with open(log_filename, 'a') as f:
            f.write(f"\n\nTraining completed in {time.time() - train_start_time:.2f} seconds\n")
            f.write(f"Model directory: {model_dir}\n")
            
            if best_model_state is not None:
                f.write(f"Best validation loss: {best_val_loss:.4f} at epoch {best_model_state['epoch']+1}\n")
                f.write(f"Best model saved to: {os.path.join(model_dir, 'model_best.pt')}\n")
            f.write(f"Train loss: {best_model_state['train_loss']:.4f}, Validation loss: {best_model_state['val_loss']:.4f}\n")
            f.write(f"Train TL: {best_model_state['train_tl']:.4f}, Validation TL: {best_model_state['val_tl']:.4f}\n")
            
            f.write(f"Final model saved to: {last_model_path}\n")

    print(f"\nResults have been saved to {log_filename}")
    print(f"Model checkpoints saved to {model_dir}")
    print(f"- Best model: {os.path.join(model_dir, 'model_best.pt')}")
    print(f"- Last model: {os.path.join(model_dir, 'model_last.pt')}")

# ---------------------------
# Set-Transformer based model
# ---------------------------

class SetEncoder(nn.Module):
    """Encodes a set of N vectors (shape B×N×n_features) into equivariant hidden representations.

    It simply embeds each element with a linear layer and then applies several
    permutation-equivariant Transformer encoder layers (a Set Transformer SAB stack).
    """

    def __init__(self, n_features: int, d_model: int = 128, n_heads: int = 4, n_layers: int = 3):
        super().__init__()

        # Initial element-wise embedding from raw n-dim vector to d_model
        self.input_proj = nn.Linear(n_features, d_model)

        # A standard Transformer encoder layer is already permutation-equivariant over the set
        encoder_layer = TransformerEncoderLayer(d_model, n_heads)
        self.encoder = TransformerEncoder(encoder_layer, n_layers)

    def forward(self, X):
        # X : (batch, N, n_features)
        H = self.input_proj(X)          # (batch, N, d_model)
        H = self.encoder(H)             # (batch, N, d_model)
        return H

class SetTransformerQAP(nn.Module):
    """Permutation-equivariant model for QAP based on two Set encoders and a pairwise MLP.

    Given distance matrix A (locations) and flow matrix B (facilities), each of
    shape (batch, N, N), the model returns a doubly-stochastic assignment matrix
    of shape (batch, N, N).  The pipeline is:

        1. Row vectors of A and B  → Set Encoders  → context-enriched embeddings
        2. All pair combinations concatenated  → small MLP  → non-negative score
        3. Lightweight row/column normalization to approximate doubly-stochastic constraints
    """

    def __init__(self, n: int, d_model: int = 128, n_heads: int = 4, n_layers: int = 3, hidden_pair: int = 128):
        super().__init__()

        self.n = n

        # Two independent encoders – one for facilities (B), one for locations (A)
        self.fac_encoder = SetEncoder(n, d_model, n_heads, n_layers)
        self.loc_encoder = SetEncoder(n, d_model, n_heads, n_layers)
        
        # Encoder for best_perm matrix (optional input)
        self.perm_encoder = SetEncoder(n, d_model, n_heads, n_layers)

        # MLP that converts concatenated pair embeddings to an intermediate representation
        self.pair_mlp = nn.Sequential(
            nn.Linear(3 * d_model, hidden_pair),  # Changed from 2*d_model to 3*d_model to include perm embedding
            nn.ReLU(inplace=True),
        )
        
        # Final layer that maps pairwise MLP output to a positive score
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_pair, 1),
            nn.Softplus()  # ensures positivity which helps IPF
        )

    def forward(self, A, B, best_perm=None, ipf_iters: int = 10):
        """A, B: (batch, N, N), best_perm: (batch, N, N) or None"""
        batch_size = A.size(0)

        # 1. Encode rows as set elements ---------------------------------------------------
        F_emb = self.fac_encoder(B)      # (batch, N, d)
        L_emb = self.loc_encoder(A)      # (batch, N, d)
        
        # Initialize P_emb with zeros if best_perm is None
        if best_perm is None:
            # Create a uniform distribution if no best_perm is provided
            device = A.device
            best_perm = torch.ones(batch_size, self.n, self.n, device=device) / self.n
        
        P_emb = self.perm_encoder(best_perm)  # (batch, N, d)

        # 2. Compute pairwise scores (outer concatenation) --------------------------------
        # Expand & concat to shape (batch, N, N, 3d)
        F_exp = F_emb.unsqueeze(2).expand(-1, -1, self.n, -1)
        L_exp = L_emb.unsqueeze(1).expand(-1, self.n, -1, -1)
        P_exp = P_emb.unsqueeze(1).expand(-1, self.n, -1, -1)
        
        # Concatenate all three embeddings
        pair_feat = torch.cat([F_exp, L_exp, P_exp], dim=-1)

        # Get intermediate features from the pairwise MLP
        mlp_features = self.pair_mlp(pair_feat)    # (batch, N, N, hidden_pair)
        
        # Map features to positive scores (no inductive bias)
        scores = self.final_layer(mlp_features).squeeze(-1)  # (batch, N, N)
        
        # 3. Lightweight row/column normalization to approximate doubly-stochastic constraints
        # Ensure non-negativity (Softplus already positive but safeguard)
        W = torch.relu(scores)  # (batch, N, N)

        # Row normalisation: each row sums to 1
        row_sums = W.sum(dim=2, keepdim=True).clamp(min=1e-8)
        W = W / row_sums

        # Column normalisation: each column sums to 1 (approximately after previous step)
        col_sums = W.sum(dim=1, keepdim=True).clamp(min=1e-8)
        W = W / col_sums

        return W

if __name__ == "__main__":
    main() 