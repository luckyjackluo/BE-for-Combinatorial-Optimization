#!/usr/bin/env python3
"""
Different step size strategies for PGD optimization to handle large gradient norms.
These functions can be used to replace the step size calculation in test_grad_be_qap.py
"""

import torch

def compute_step_size_log_scaling(grad_norm, lr):
    """
    Option 1: Log scaling to prevent extremely small step sizes
    Good for very large gradient norms
    """
    if grad_norm > 0:
        return lr / (1 + torch.log(1 + grad_norm))
    else:
        return lr

def compute_step_size_sqrt_scaling(grad_norm, lr):
    """
    Option 2: Square root scaling
    Moderate scaling for large gradients
    """
    if grad_norm > 0:
        return lr / (1 + torch.sqrt(grad_norm))
    else:
        return lr

def compute_step_size_clipped(grad_norm, lr, max_grad_norm=100.0):
    """
    Option 3: Clipped gradient norm
    Caps the gradient norm to prevent extremely small step sizes
    """
    if grad_norm > 0:
        clipped_norm = torch.clamp(grad_norm, max=max_grad_norm)
        return lr / (1 + clipped_norm)
    else:
        return lr

def compute_step_size_adaptive_clipping(grad_norm, lr, min_step_size=1e-6):
    """
    Option 4: Adaptive clipping with minimum step size
    Ensures step size never goes below a minimum threshold
    """
    if grad_norm > 0:
        step_size = lr / (1 + grad_norm)
        return torch.clamp(step_size, min=min_step_size)
    else:
        return lr

def compute_step_size_exponential_decay(grad_norm, lr, decay_factor=0.1):
    """
    Option 5: Exponential decay scaling
    Uses exponential function for smoother scaling
    """
    if grad_norm > 0:
        return lr * torch.exp(-decay_factor * grad_norm)
    else:
        return lr

def compute_step_size_robust_scaling(grad_norm, lr, alpha=0.5):
    """
    Option 6: Robust scaling (Huber-like)
    Combines linear and log scaling for robustness
    """
    if grad_norm > 0:
        if grad_norm <= 1.0:
            # Linear scaling for small gradients
            return lr / (1 + alpha * grad_norm)
        else:
            # Log scaling for large gradients
            return lr / (1 + alpha * (1 + torch.log(grad_norm)))
    else:
        return lr

def compute_step_size_momentum_adaptive(grad_norm, lr, momentum_buffer, beta=0.9):
    """
    Option 7: Momentum-adaptive step size
    Adjusts step size based on momentum buffer norm
    """
    if grad_norm > 0:
        # Consider both current gradient and momentum
        momentum_norm = torch.norm(momentum_buffer) if momentum_buffer is not None else 0.0
        combined_norm = torch.sqrt(grad_norm**2 + beta**2 * momentum_norm**2)
        return lr / (1 + torch.log(1 + combined_norm))
    else:
        return lr

# Example usage and comparison
def compare_step_size_methods(grad_norm, lr=0.01):
    """
    Compare different step size methods for a given gradient norm
    """
    print(f"Gradient norm: {grad_norm:.2f}")
    print(f"Learning rate: {lr}")
    print("-" * 50)
    
    methods = [
        ("Original (1 + grad_norm)", lambda: lr / (1 + grad_norm)),
        ("Log scaling", lambda: compute_step_size_log_scaling(grad_norm, lr)),
        ("Sqrt scaling", lambda: compute_step_size_sqrt_scaling(grad_norm, lr)),
        ("Clipped (max=100)", lambda: compute_step_size_clipped(grad_norm, lr, 100.0)),
        ("Adaptive clipping (min=1e-6)", lambda: compute_step_size_adaptive_clipping(grad_norm, lr, 1e-6)),
        ("Exponential decay", lambda: compute_step_size_exponential_decay(grad_norm, lr, 0.1)),
        ("Robust scaling", lambda: compute_step_size_robust_scaling(grad_norm, lr, 0.5)),
    ]
    
    for name, method in methods:
        try:
            step_size = method()
            print(f"{name:30s}: {step_size:.8f}")
        except Exception as e:
            print(f"{name:30s}: Error - {e}")

if __name__ == "__main__":
    # Test with different gradient norm values
    test_norms = [1.0, 10.0, 100.0, 1000.0, 10000.0]
    
    for norm in test_norms:
        print(f"\n{'='*60}")
        compare_step_size_methods(norm)
        print(f"{'='*60}") 