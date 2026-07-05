import torch
from . import _C

class DifferentiablePDist(torch.autograd.Function):
    @staticmethod
    def forward(ctx, point_cloud, deadzone):
        dist_matrix = _C.compute_pdist(point_cloud)
        ctx.save_for_backward(point_cloud, dist_matrix)
        ctx.deadzone = deadzone
        return dist_matrix

    @staticmethod
    def backward(ctx, grad_dist):
        point_cloud, dist_matrix = ctx.saved_tensors
        grad_dist = grad_dist.contiguous()
        grad_point_cloud = _C.compute_pdist_backward(
            grad_dist, point_cloud, dist_matrix, ctx.deadzone
        )
        return grad_point_cloud, None

class DifferentiableRadiusGraph(torch.autograd.Function):
    @staticmethod
    def forward(ctx, point_cloud, radius, max_edges, deadzone):
        indices, values = _C.compute_radius_graph(point_cloud, radius, max_edges)
        
        ctx.save_for_backward(indices, point_cloud, values)
        ctx.deadzone = deadzone
        
        # [CRITICAL FIX] Mark discrete tensors as non-differentiable
        ctx.mark_non_differentiable(indices)
        
        return indices, values

    @staticmethod
    def backward(ctx, grad_indices, grad_values):
        indices, point_cloud, values = ctx.saved_tensors
        grad_values = grad_values.contiguous()
        
        grad_point_cloud = _C.compute_radius_graph_backward(
            grad_values, indices, point_cloud, values, ctx.deadzone
        )
        
        return grad_point_cloud, None, None, None

# ==========================================
# PUBLIC API EXPORTS
# ==========================================

def pdist(point_cloud, deadzone=1e-5):
    """
    Computes pairwise Euclidean distances (L2) with numerical stability.
    Supports both (N, D) and (B, N, D) tensors.
    """
    return DifferentiablePDist.apply(point_cloud, deadzone)

def radius_graph(point_cloud, radius, max_edges=500000, deadzone=1e-5):
    """
    Computes a sparse radius graph for Topological Data Analysis.
    Only returns connections where distance <= radius.
    """
    assert point_cloud.dim() == 2, "radius_graph currently supports 2D (N, D) tensors."
    return DifferentiableRadiusGraph.apply(point_cloud.contiguous(), radius, max_edges, deadzone)