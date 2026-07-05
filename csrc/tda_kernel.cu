#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ============================================================================
// DENSE DISTANCE - CUDA KERNELS
// ============================================================================
__global__ void pdist_forward_batched_kernel(
    const float* __restrict__ pc_data,
    float* __restrict__ dist_data,
    int B, int N, int D) {
    
    int b = blockIdx.z; 
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y; 

    if (b < B && i < N && j < N) {
        float sq_dist = 0.0f;
        
        const float* batch_pc = pc_data + (b * N * D);
        float* batch_dist = dist_data + (b * N * N);
        
        for (int k = 0; k < D; ++k) {
            float diff = batch_pc[i * D + k] - batch_pc[j * D + k];
            sq_dist += diff * diff;
        }
        
        batch_dist[i * N + j] = sqrtf(sq_dist);
    }
}

torch::Tensor compute_pdist_cuda(const torch::Tensor& point_cloud) {
    TORCH_CHECK(point_cloud.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(point_cloud.is_contiguous(), "Input must be contiguous");
    
    const int B = point_cloud.size(0);
    const int N = point_cloud.size(1);
    const int D = point_cloud.size(2);
    
    auto dist_matrix = torch::empty({B, N, N}, point_cloud.options());
    
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y, B);
    
    pdist_forward_batched_kernel<<<blocks, threads>>>(
        point_cloud.data_ptr<float>(),
        dist_matrix.data_ptr<float>(),
        B, N, D
    );
    
    return dist_matrix;
}

__global__ void pdist_backward_batched_kernel(
    const float* __restrict__ grad_dist,
    const float* __restrict__ pc_data,
    const float* __restrict__ dist_data,
    float* __restrict__ grad_pc_data,
    int B, int N, int D,
    float deadzone) { 
    
    int b = blockIdx.z; 
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int k = blockIdx.y * blockDim.y + threadIdx.y; 

    if (b < B && i < N && k < D) {
        float grad_accum = 0.0f;
        const float epsilon = 1e-8f; 

        const float* b_grad_dist = grad_dist + (b * N * N);
        const float* b_pc = pc_data + (b * N * D);
        const float* b_dist = dist_data + (b * N * N);
        float* b_grad_pc = grad_pc_data + (b * N * D);

        for (int j = 0; j < N; ++j) {
            if (i == j) continue;

            float d = b_dist[i * N + j];
            if (d < deadzone) continue;

            float g_sym = b_grad_dist[i * N + j] + b_grad_dist[j * N + i];

            if (g_sym != 0.0f) {
                float diff = b_pc[i * D + k] - b_pc[j * D + k];
                grad_accum += g_sym * (diff / (d + epsilon));
            }
        }
        
        b_grad_pc[i * D + k] = grad_accum;
    }
}

torch::Tensor compute_pdist_backward_cuda(
    const torch::Tensor& grad_dist,
    const torch::Tensor& point_cloud,
    const torch::Tensor& dist_matrix,
    float deadzone) {
    
    TORCH_CHECK(grad_dist.is_cuda() && point_cloud.is_cuda(), "Inputs must be on CUDA");
    
    const int B = point_cloud.size(0);
    const int N = point_cloud.size(1);
    const int D = point_cloud.size(2);
    
    auto grad_point_cloud = torch::zeros_like(point_cloud);
    
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (D + threads.y - 1) / threads.y, B);
    
    pdist_backward_batched_kernel<<<blocks, threads>>>(
        grad_dist.data_ptr<float>(),
        point_cloud.data_ptr<float>(),
        dist_matrix.data_ptr<float>(),
        grad_point_cloud.data_ptr<float>(),
        B, N, D, deadzone
    );
    
    return grad_point_cloud;
}

// ============================================================================
// SPARSE RADIUS GRAPH - CUDA KERNELS
// ============================================================================
__global__ void radius_graph_forward_kernel(
    const float* __restrict__ pc_data,
    int64_t* __restrict__ indices_data, 
    float* __restrict__ values_data,    
    int* __restrict__ count_data,       
    int N, int D, 
    float radius, 
    int max_edges) {

    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y; 

    if (i < N && j < N && i != j) {
        float sq_dist = 0.0f;
        
        for (int k = 0; k < D; ++k) {
            float diff = pc_data[i * D + k] - pc_data[j * D + k];
            sq_dist += diff * diff;
        }
        
        float dist = sqrtf(sq_dist);

        if (dist <= radius) {
            int idx = atomicAdd(count_data, 1);
            
            if (idx < max_edges) {
                indices_data[0 * max_edges + idx] = i; 
                indices_data[1 * max_edges + idx] = j; 
                values_data[idx] = dist;
            }
        }
    }
}

std::vector<torch::Tensor> compute_radius_graph_cuda(
    const torch::Tensor& point_cloud, 
    float radius, 
    int max_edges) {
    
    TORCH_CHECK(point_cloud.is_cuda(), "point_cloud must be on CUDA");
    
    const int N = point_cloud.size(0);
    const int D = point_cloud.size(1);
    
    auto options_int = point_cloud.options().dtype(torch::kInt64);
    auto options_float = point_cloud.options();
    
    auto indices = torch::empty({2, max_edges}, options_int);
    auto values = torch::empty({max_edges}, options_float);
    
    auto counter = torch::zeros({1}, point_cloud.options().dtype(torch::kInt32));
    
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);
    
    radius_graph_forward_kernel<<<blocks, threads>>>(
        point_cloud.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        values.data_ptr<float>(),
        counter.data_ptr<int>(),
        N, D, radius, max_edges
    );
    
    int actual_edges = counter.item<int>();
    
    if (actual_edges > max_edges) {
        TORCH_WARN("Radius graph exceeded max_edges. Some topological connections were dropped.");
        actual_edges = max_edges;
    }
    
    auto valid_indices = indices.index({torch::indexing::Slice(), torch::indexing::Slice(0, actual_edges)}).contiguous();
    auto valid_values = values.index({torch::indexing::Slice(0, actual_edges)}).contiguous();
    
    return {valid_indices, valid_values};
}

__global__ void radius_graph_backward_kernel(
    const float* __restrict__ grad_values,
    const int64_t* __restrict__ indices,
    const float* __restrict__ pc_data,
    const float* __restrict__ values_data,
    float* __restrict__ grad_pc_data,
    int E, int D, float deadzone) {

    int e = blockIdx.x * blockDim.x + threadIdx.x;

    if (e < E) {
        int i = indices[0 * E + e]; 
        int j = indices[1 * E + e]; 

        float d = values_data[e];
        
        if (d < deadzone) return;

        float g = grad_values[e];
        const float epsilon = 1e-8f;

        for (int k = 0; k < D; ++k) {
            float diff = pc_data[i * D + k] - pc_data[j * D + k];
            float grad_contrib = g * (diff / (d + epsilon));

            atomicAdd(&grad_pc_data[i * D + k], grad_contrib);
            atomicAdd(&grad_pc_data[j * D + k], -grad_contrib);
        }
    }
}

torch::Tensor compute_radius_graph_backward_cuda(
    const torch::Tensor& grad_values,
    const torch::Tensor& indices,
    const torch::Tensor& point_cloud,
    const torch::Tensor& values_data,
    float deadzone) {
    
    TORCH_CHECK(grad_values.is_cuda(), "grad_values must be on CUDA");
    
    const int E = grad_values.size(0); 
    const int N = point_cloud.size(0);
    const int D = point_cloud.size(1);
    
    auto grad_point_cloud = torch::zeros_like(point_cloud);
    
    if (E == 0) return grad_point_cloud;
    
    dim3 threads(256);
    dim3 blocks((E + threads.x - 1) / threads.x); 
    
    radius_graph_backward_kernel<<<blocks, threads>>>(
        grad_values.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        point_cloud.data_ptr<float>(),
        values_data.data_ptr<float>(),
        grad_point_cloud.data_ptr<float>(),
        E, D, deadzone
    );
    
    return grad_point_cloud;
}