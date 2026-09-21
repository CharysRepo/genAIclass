import torch
import time
import numpy as np
from tabulate import tabulate
import os
from numba import cuda

CC_CORES_PER_SM = {
    (2,0) : 32, (2,1) : 48, (3,0) : 192, (3,5) : 192, (3,7) : 192, (5,0) : 128,
    (5,2) : 128, (6,0) : 64, (6,1) : 128, (7,0) : 64, (7,5) : 64, (8,0) : 64,
    (8,6) : 128, (8,9) : 128, (9,0) : 128, (10,0) : 128, (12,0) : 128
}

def get_cuda_cores(gpu_index=0):
    # Select the device for numba
    cuda.select_device(gpu_index)
    device = cuda.get_current_device()

    # Get SM count and compute capability
    sms = getattr(device, 'MULTIPROCESSOR_COUNT')
    cc_major, cc_minor = device.compute_capability

    cores_per_sm = CC_CORES_PER_SM.get((cc_major, cc_minor))

    if cores_per_sm is None:
        return f"Unknown compute capability: {cc_major}.{cc_minor}. Core count not in dictionary."

    total_cores = cores_per_sm * sms
    return total_cores

def benchmark_matmul(size, device, num_runs=10):
    """
    Benchmark matrix multiplication for a given size and device
    """
    # Create random matrices on the specified device
    matrix1 = torch.randn(size, size, device=device)
    matrix2 = torch.randn(size, size, device=device)

    # Warmup run
    torch.matmul(matrix1, matrix2)
    if device == 'cuda':
        torch.cuda.synchronize() # Wait for GPU to finish

    # Timing runs
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        torch.matmul(matrix1, matrix2)
        if device == 'cuda':
            torch.cuda.synchronize() # Wait for GPU to finish
        end = time.perf_counter()
        times.append(end - start)
    
    return np.mean(times)

def run_benchmarks():
    # Test different matrix sizes
    sizes = [128, 512, 2048, 8192, 12288]
    results = []

    for size in sizes:
        cpu_time = benchmark_matmul(size, 'cpu')
        
        if torch.cuda.is_available():
            gpu_time = benchmark_matmul(size, 'cuda')
            speedup = cpu_time / gpu_time
            results.append([f"{size}x{size}", f"{cpu_time:.6f}s", f"{gpu_time:.6f}s", f"{speedup:.2f}x"])
        else:
            results.append([f"{size}x{size}", f"{cpu_time:.6f}s", "N/A", "N/A"])
    
    print(tabulate(results, headers=["Matrix Size", "CPU Time (Avg)", "GPU Time (Avg)", "GPU Speedup"]))

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("GPU (CUDA) is available. Benchmarking both CPU and GPU.")
        print("CPU Cores:", os.cpu_count())
        gpu_count = torch.cuda.device_count()
        print(f"Total GPUs detected: {gpu_count}")
        for i in range(gpu_count):
            print(f"GPU {i} Name: {torch.cuda.get_device_name(i)}")
            cores = get_cuda_cores(i)
            print(f"GPU {i} CUDA Cores: {cores}")
    else:
        print("GPU (CUDA) not available. Benchmarking CPU only.")
        print("Install a CUDA-enabled PyTorch version to test GPU performance.")
        print("CPU Cores:", os.cpu_count())

    run_benchmarks()
