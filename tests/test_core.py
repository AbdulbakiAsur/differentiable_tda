# tests/test_core.py
import torch
import differentiable_tda

# Fix seed for reproducibility
torch.manual_seed(42)

def test_forward_pass_accuracy():
    """Verify that custom C++ forward pass matches PyTorch native cdist."""
    x = torch.rand((5, 3), dtype=torch.float32)
    
    dist_custom = differentiable_tda.pdist(x)
    dist_native = torch.cdist(x, x, p=2.0)
    
    # Assert will fail the test if the condition is False
    assert torch.allclose(dist_custom, dist_native, atol=1e-5), "Forward pass mismatch!"

def test_backward_pass_accuracy():
    """Verify that custom CUDA/CPU backward pass gradients match PyTorch."""
    x_custom = torch.rand((5, 3), dtype=torch.float32, requires_grad=True)
    x_native = x_custom.clone().detach().requires_grad_(True)
    
    loss_custom = differentiable_tda.pdist(x_custom).sum()
    loss_custom.backward()
    
    loss_native = torch.cdist(x_native, x_native, p=2.0).sum()
    loss_native.backward()
    
    assert torch.allclose(x_custom.grad, x_native.grad, atol=1e-4), "Backward gradients mismatch!"

def test_numerical_stability_deadzone():
    """Ensure that identical topological points do not cause gradient explosions (NaN)."""
    # Force GPU usage for deadzone testing
    if not torch.cuda.is_available():
        return # Skip if no GPU
        
    x_anomaly = torch.tensor([
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0] 
    ], dtype=torch.float32, requires_grad=True, device='cuda')
    
    loss = differentiable_tda.pdist(x_anomaly).sum()
    loss.backward()
    
    # Assert no NaNs are present in the gradient
    assert not torch.isnan(x_anomaly.grad).any(), "Gradient explosion detected (NaNs present)!"