#include <torch/extension.h>
#include <cmath>
#include <vector>

// ============================================================================
// DYNAMIC CUDA DECLARATIONS (Compiled only if WITH_CUDA is defined)
// ============================================================================
#ifdef WITH_CUDA

torch::Tensor compute_pdist_cuda(const torch::Tensor& point_cloud);

torch::Tensor compute_pdist_backward_cuda(
    const torch::Tensor& grad_dist,
    const torch::Tensor& point_cloud,
    const torch::Tensor& dist_matrix,
    float deadzone);

std::vector<torch::Tensor> compute_radius_graph_cuda(
    const torch::Tensor& point_cloud, 
    float radius, 
    int max_edges);

torch::Tensor compute_radius_graph_backward_cuda(
    const torch::Tensor& grad_values, 
    const torch::Tensor& indices,
    const torch::Tensor& point_cloud, 
    const torch::Tensor& values_data, 
    float deadzone);

#endif // WITH_CUDA

// ============================================================================
// DENSE DISTANCE - CPU FALLBACK
// ============================================================================
torch::Tensor compute_pdist_cpu(const torch::Tensor& point_cloud) {
    TORCH_CHECK(point_cloud.dim() == 2, "Input must be a 2D tensor (N, D)");
    TORCH_CHECK(point_cloud.scalar_type() == torch::kFloat32, "Input must be Float32");
    TORCH_CHECK(point_cloud.is_contiguous(), "Memory layout must be contiguous");

    const int64_t N = point_cloud.size(0);
    const int64_t D = point_cloud.size(1);

    auto dist_matrix = torch::empty({N, N}, point_cloud.options());
    const float* pc_data = point_cloud.data_ptr<float>();
    float* dist_data = dist_matrix.data_ptr<float>();

    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sq_dist = 0.0f;
            for (int64_t k = 0; k < D; ++k) {
                float diff = pc_data[i * D + k] - pc_data[j * D + k];
                sq_dist += diff * diff;
            }
            dist_data[i * N + j] = std::sqrt(sq_dist);
        }
    }
    return dist_matrix;
}

torch::Tensor compute_pdist_backward_cpu(
    const torch::Tensor& grad_dist,
    const torch::Tensor& point_cloud,
    const torch::Tensor& dist_matrix, 
    float deadzone) {

    const int64_t N = point_cloud.size(0);
    const int64_t D = point_cloud.size(1);

    auto grad_point_cloud = torch::zeros({N, D}, point_cloud.options());
    const float* grad_dist_data = grad_dist.data_ptr<float>();
    const float* pc_data = point_cloud.data_ptr<float>();
    const float* dist_data = dist_matrix.data_ptr<float>();
    float* grad_pc_data = grad_point_cloud.data_ptr<float>();
    const float epsilon = 1e-8f;

    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            if (i == j) continue; 
            float d = dist_data[i * N + j];
            if (d < deadzone) continue;
            float g = grad_dist_data[i * N + j] + grad_dist_data[j * N + i];
            if (g != 0.0f) {
                for (int64_t k = 0; k < D; ++k) {
                    float diff = pc_data[i * D + k] - pc_data[j * D + k];
                    grad_pc_data[i * D + k] += g * (diff / (d + epsilon));
                }
            }
        }
    }
    return grad_point_cloud;
}

// ============================================================================
// SPARSE RADIUS GRAPH - CPU FALLBACK
// ============================================================================
std::vector<torch::Tensor> compute_radius_graph_cpu(
    const torch::Tensor& point_cloud, float radius, int max_edges) {
    const int N = point_cloud.size(0);
    const int D = point_cloud.size(1);
    auto options_int = point_cloud.options().dtype(torch::kInt64);
    auto options_float = point_cloud.options();
    
    auto indices = torch::empty({2, max_edges}, options_int);
    auto values = torch::empty({max_edges}, options_float);
    
    int64_t* idx_ptr = indices.data_ptr<int64_t>();
    float* val_ptr = values.data_ptr<float>();
    const float* pc_data = point_cloud.data_ptr<float>();
    int current_edge_count = 0;
    
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == j) continue;
            float sq_dist = 0.0f;
            for (int k = 0; k < D; ++k) {
                float diff = pc_data[i * D + k] - pc_data[j * D + k];
                sq_dist += diff * diff;
            }
            float dist = std::sqrt(sq_dist);
            if (dist <= radius) {
                if (current_edge_count < max_edges) {
                    idx_ptr[0 * max_edges + current_edge_count] = i;
                    idx_ptr[1 * max_edges + current_edge_count] = j;
                    val_ptr[current_edge_count] = dist;
                }
                current_edge_count++;
            }
        }
    }
    if (current_edge_count > max_edges) {
        TORCH_WARN("CPU Radius graph exceeded max_edges. Some connections were dropped.");
        current_edge_count = max_edges;
    }
    auto valid_indices = indices.index({torch::indexing::Slice(), torch::indexing::Slice(0, current_edge_count)}).contiguous();
    auto valid_values = values.index({torch::indexing::Slice(0, current_edge_count)}).contiguous();
    return {valid_indices, valid_values};
}

torch::Tensor compute_radius_graph_backward_cpu(
    const torch::Tensor& grad_values, const torch::Tensor& indices,
    const torch::Tensor& point_cloud, const torch::Tensor& values_data, float deadzone) {
    const int E = grad_values.size(0);
    const int N = point_cloud.size(0);
    const int D = point_cloud.size(1);
    
    auto grad_point_cloud = torch::zeros_like(point_cloud);
    if (E == 0) return grad_point_cloud;
    
    const float* grad_v_ptr = grad_values.data_ptr<float>();
    const int64_t* idx_ptr = indices.data_ptr<int64_t>();
    const float* pc_ptr = point_cloud.data_ptr<float>();
    const float* val_ptr = values_data.data_ptr<float>();
    float* grad_pc_ptr = grad_point_cloud.data_ptr<float>();
    
    const float epsilon = 1e-8f;
    for (int e = 0; e < E; ++e) {
        int i = idx_ptr[0 * E + e];
        int j = idx_ptr[1 * E + e];
        float d = val_ptr[e];
        if (d < deadzone) continue;
        float g = grad_v_ptr[e];
        for (int k = 0; k < D; ++k) {
            float diff = pc_ptr[i * D + k] - pc_ptr[j * D + k];
            float grad_contrib = g * (diff / (d + epsilon));
            grad_pc_ptr[i * D + k] += grad_contrib;
            grad_pc_ptr[j * D + k] -= grad_contrib;
        }
    }
    return grad_point_cloud;
}

// ============================================================================
// SMART DISPATCHERS
// ============================================================================
torch::Tensor compute_pdist(const torch::Tensor& point_cloud) {
    auto pc_contig = point_cloud.contiguous();
    TORCH_CHECK(pc_contig.dim() == 2 || pc_contig.dim() == 3, "Input must be 2D or 3D");

    if (pc_contig.is_cuda()) {
#ifdef WITH_CUDA
        bool is_2d = (pc_contig.dim() == 2);
        auto pc_3d = is_2d ? pc_contig.unsqueeze(0) : pc_contig;
        auto dist_matrix_3d = compute_pdist_cuda(pc_3d);
        return is_2d ? dist_matrix_3d.squeeze(0) : dist_matrix_3d;
#else
        TORCH_CHECK(false, "Not compiled with CUDA support.");
#endif
    } else {
        // [CRITICAL FIX]: Route 2D tensors to the CPU fallback implementation.
        if (pc_contig.dim() == 2) {
            return compute_pdist_cpu(pc_contig);
        } else {
            TORCH_CHECK(false, "Batched (3D) CPU execution is not implemented. Please use 2D tensors for CPU.");
        }
    }
}

torch::Tensor compute_pdist_backward(
    const torch::Tensor& grad_dist, const torch::Tensor& point_cloud,
    const torch::Tensor& dist_matrix, float deadzone) {
    
    auto pc_contig = point_cloud.contiguous();
    
    if (pc_contig.is_cuda()) {
#ifdef WITH_CUDA
        bool is_2d = (pc_contig.dim() == 2);
        auto grad_dist_3d = is_2d ? grad_dist.unsqueeze(0) : grad_dist.contiguous();
        auto pc_3d = is_2d ? pc_contig.unsqueeze(0) : pc_contig;
        auto dist_matrix_3d = is_2d ? dist_matrix.unsqueeze(0) : dist_matrix.contiguous();
        
        auto grad_pc_3d = compute_pdist_backward_cuda(grad_dist_3d, pc_3d, dist_matrix_3d, deadzone);
        return is_2d ? grad_pc_3d.squeeze(0) : grad_pc_3d;
#else
        TORCH_CHECK(false, "Not compiled with CUDA support.");
#endif
    } else {
        // [CRITICAL FIX]: Route 2D tensors to the CPU fallback backward implementation.
        if (pc_contig.dim() == 2) {
            return compute_pdist_backward_cpu(grad_dist.contiguous(), pc_contig, dist_matrix.contiguous(), deadzone);
        } else {
            TORCH_CHECK(false, "Batched (3D) CPU backward is not implemented. Please use 2D tensors for CPU.");
        }
    }
}

std::vector<torch::Tensor> compute_radius_graph(
    const torch::Tensor& point_cloud, float radius, int max_edges) {
    auto pc_contig = point_cloud.contiguous();
    TORCH_CHECK(pc_contig.dim() == 2, "Radius graph only supports 2D inputs.");
    
    if (pc_contig.is_cuda()) {
#ifdef WITH_CUDA
        return compute_radius_graph_cuda(pc_contig, radius, max_edges);
#else
        TORCH_CHECK(false, "Not compiled with CUDA support.");
#endif
    } else {
        return compute_radius_graph_cpu(pc_contig, radius, max_edges);
    }
}

torch::Tensor compute_radius_graph_backward(
    const torch::Tensor& grad_values, const torch::Tensor& indices,
    const torch::Tensor& point_cloud, const torch::Tensor& values_data, float deadzone) {
    auto pc_contig = point_cloud.contiguous();
    TORCH_CHECK(pc_contig.dim() == 2, "Radius graph only supports 2D inputs.");
    
    if (pc_contig.is_cuda()) {
#ifdef WITH_CUDA
        return compute_radius_graph_backward_cuda(grad_values.contiguous(), indices.contiguous(), pc_contig, values_data.contiguous(), deadzone);
#else
        TORCH_CHECK(false, "Not compiled with CUDA support.");
#endif
    } else {
        return compute_radius_graph_backward_cpu(grad_values.contiguous(), indices.contiguous(), pc_contig, values_data.contiguous(), deadzone);
    }
}

// ============================================================================
// PYTHON BINDINGS
// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_pdist", &compute_pdist, "Smart Pairwise Distance");
    m.def("compute_pdist_backward", &compute_pdist_backward, "Smart Backward");
    m.def("compute_radius_graph", &compute_radius_graph, "Sparse Radius Graph");
    m.def("compute_radius_graph_backward", &compute_radius_graph_backward, "Sparse Radius Graph Backward");
}