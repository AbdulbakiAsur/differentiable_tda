import torch
import differentiable_tda as dtda
import gc

def benchmark_sparse_vs_dense():
    print("="*75)
    print("🚀 HARDWARE BENCHMARK: SPARSE VS DENSE TDA (EXTREME SCALING)")
    print("="*75)
    
    # Warmup
    dummy = torch.rand((100, 64), device='cuda')
    _ = dtda.pdist(dummy)
    torch.cuda.synchronize()

    # N'i 100,000'e kadar zorluyoruz. PyTorch yüksek ihtimalle 30-40k'da VRAM OOM yiyecektir.
    test_cases = [1000, 5000, 10000, 20000, 40000]
    D = 64
    radius = 2.5 # Curse of dimensionality gereği 2.5 makul bir eşik
    
    print(f"{'N Points':<10} | {'Backend':<15} | {'Time (ms)':<15} | {'VRAM (MB)':<15} | {'Status'}")
    print("-" * 75)

    for N in test_cases:
        x = torch.rand((N, D), dtype=torch.float32, device='cuda', requires_grad=True)
        
        # ---------------------------------------------------------
        # 1. NATIVE PYTORCH (DENSE)
        # ---------------------------------------------------------
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()
        
        status_native = "SUCCESS"
        time_native = 0.0
        vram_native = 0.0
        
        try:
            start_native = torch.cuda.Event(enable_timing=True)
            end_native = torch.cuda.Event(enable_timing=True)
            
            start_native.record()
            dist_native = torch.cdist(x, x, p=2.0, compute_mode='donot_use_mm_for_euclid_dist')
            end_native.record()
            torch.cuda.synchronize()
            
            time_native = start_native.elapsed_time(end_native)
            vram_native = torch.cuda.max_memory_allocated() / (1024 ** 2)
            del dist_native
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                status_native = "OOM CRASH 💥"
            else:
                status_native = "FAILED"
        
        time_str = f"{time_native:.2f}" if status_native == "SUCCESS" else "N/A"
        vram_str = f"{vram_native:.2f}" if status_native == "SUCCESS" else "OOM"
        print(f"{N:<10} | {'PyTorch Dense':<15} | {time_str:<15} | {vram_str:<15} | {status_native}")
        
        # ---------------------------------------------------------
        # 2. CUSTOM CUDA KERNEL (SPARSE)
        # ---------------------------------------------------------
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()
        
        status_custom = "SUCCESS"
        time_custom = 0.0
        vram_custom = 0.0
        
        try:
            start_custom = torch.cuda.Event(enable_timing=True)
            end_custom = torch.cuda.Event(enable_timing=True)
            
            start_custom.record()
            # Dinamik buffer: N büyüdükçe kenar olasılığı artar
            indices, values = dtda.radius_graph(x, radius=radius, max_edges=N * 100)
            end_custom.record()
            torch.cuda.synchronize()
            
            time_custom = start_custom.elapsed_time(end_custom)
            vram_custom = torch.cuda.max_memory_allocated() / (1024 ** 2)
            del indices, values
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                status_custom = "OOM CRASH 💥"
            else:
                status_custom = "FAILED"
        
        time_str = f"{time_custom:.2f}" if status_custom == "SUCCESS" else "N/A"
        vram_str = f"{vram_custom:.2f}" if status_custom == "SUCCESS" else "OOM"
        print(f"{N:<10} | {'Custom Sparse':<15} | {time_str:<15} | {vram_str:<15} | {status_custom}")
        print("-" * 75)

if __name__ == "__main__":
    if torch.cuda.is_available():
        benchmark_sparse_vs_dense()
    else:
        print("CRITICAL: No CUDA device found. Benchmarks require a GPU.")