import numpy as np
from numba import jit, config, prange
import matplotlib.pyplot as plt
import time
import cupy as cp
from cupyx import jit as cujit

from utils import *
import warnings

# disable FutureWarning for Numba JIT
warnings.filterwarnings("ignore", category=FutureWarning)

# Wrapper functions for timing and comparing Monte Carlo Pi calculations
# -------------------------------------------------------------------------------------------------
@avg_timing_decorator
def monteCarloPiBase_AVG(num_samples):
    """Wrapper function for average timing of base Monte Carlo Pi calculation."""
    return monteCarloPiBase(num_samples)

@avg_timing_decorator
def monteCarloPiNumpy_AVG(num_samples):
    """Wrapper function for average timing of Monte Carlo Pi calculation using NumPy."""
    return monteCarloPiNumpy(num_samples)

@avg_timing_decorator
def monteCarloPiNumba_AVG(num_samples):
    """Wrapper function for average timing of Monte Carlo Pi calculation using Numba JIT."""
    return monteCarloPiNumba(num_samples)

@avg_timing_decorator
def monteCarloPiCuPy_AVG(num_samples):
    """Wrapper function for average timing of Monte Carlo Pi calculation using CuPy."""
    return monteCarloPiCuPy(num_samples)

@avg_timing_decorator
def monteCarloPiCuPyKernel_AVG(num_samples):
    """Wrapper function for average timing of Monte Carlo Pi calculation using a CuPy CUDA kernel."""
    return monteCarloPiCuPyKernel(num_samples)

@avg_timing_decorator
def monteCarloPiCython_AVG(num_samples):
    """Wrapper function for average timing of Monte Carlo Pi calculation using Cython."""
    return monteCarloPiCython(num_samples)

# Actual Monte Carlo Pi calculation functions
# -------------------------------------------------------------------------------------------------

def monteCarloPiBase(num_samples):
    """Base function for Monte Carlo Pi calculation."""
    inside_circle = 0
    for _ in range(num_samples):
        x = np.random.rand()
        y = np.random.rand()
        if x**2 + y**2 <= 1.0:
            inside_circle += 1
    pi_estimate = (inside_circle / num_samples) * 4
    return pi_estimate

def monteCarloPiNumpy(num_samples):
    """Estimate Pi using Monte Carlo method with NumPy."""
    x = np.random.rand(num_samples)
    y = np.random.rand(num_samples)
    inside_circle = (x**2 + y**2) <= 1.0
    pi_estimate = (inside_circle.sum() / num_samples) * 4
    return pi_estimate

@jit(nopython=True, parallel=True, cache=True)
def monteCarloPiNumba(num_samples):
    """Estimate Pi using Monte Carlo method with Numba JIT."""
    inside_circle = 0
    for _ in prange(num_samples):
        x = np.random.rand()
        y = np.random.rand()
        if x**2 + y**2 <= 1.0:
            inside_circle += 1
    pi_estimate = (inside_circle / num_samples) * 4
    return pi_estimate

def monteCarloPiCuPy(num_samples):
    """Estimate Pi using Monte Carlo method with CuPy."""
    x = cp.random.rand(num_samples)
    y = cp.random.rand(num_samples)
    inside_circle = (x**2 + y**2) <= 1.0
    pi_estimate = (inside_circle.sum() / num_samples) * 4
    return float(pi_estimate)

# CUDA kernel for Monte Carlo Pi estimation using CuPy RawKernel
monteCarloPiKernel = cp.RawKernel(r"""
extern "C" __global__ void monteCarloPi(
    double* x, double* y, unsigned char* inside, int num_samples) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_samples) {
        inside[tid] = (x[tid] * x[tid] + y[tid] * y[tid]) <= 1.0;
    }
}
""", "monteCarloPi")

def monteCarloPiCuPyKernel(num_samples):
    """Estimate Pi using Monte Carlo method with a custom CUDA kernel."""
    threads_per_block = 256
    blocks_per_grid = (num_samples + threads_per_block - 1) // threads_per_block
    
    x = cp.random.rand(num_samples)
    y = cp.random.rand(num_samples)
    inside = cp.zeros(num_samples, dtype=cp.uint8)
    
    monteCarloPiKernel(
        (blocks_per_grid,), (threads_per_block,),
        (x, y, inside, cp.int32(num_samples))
    )
    
    pi_estimate = (inside.sum() / num_samples) * 4
    return float(pi_estimate)

def monteCarloPiCython(num_samples):
    """Estimate Pi using Monte Carlo method with Cython."""
    # Import the Cython function from the compiled module.
    from monte_carlo_pi_cython import _monteCarloPiCython
    return _monteCarloPiCython(num_samples) 


def main():
    # Parameters
    # num_samples = 100_000_000
    num_samples = 1_000_000
    
    print(f"Estimating Pi using {num_samples:,} samples.")

    pi_base = monteCarloPiBase_AVG(num_samples)
    pi_numpy = monteCarloPiNumpy_AVG(num_samples)
    pi_numba = monteCarloPiNumba_AVG(num_samples)
    if check_cupy():
        pi_cupy = monteCarloPiCuPy_AVG(num_samples)
        pi_cupy_kernel = monteCarloPiCuPyKernel_AVG(num_samples)
    pi_cython = monteCarloPiCython_AVG(num_samples)
    
    print(f"\nPi estimate using Base:        {pi_base}")
    print(f"Pi estimate using NumPy:       {pi_numpy}")
    print(f"Pi estimate using Numba:       {pi_numba}")
    if check_cupy():
        print(f"Pi estimate using CuPy:        {pi_cupy}")
        print(f"Pi estimate using CuPy Kernel: {pi_cupy_kernel}")
    print(f"Pi estimate using Cython:      {pi_cython}")
        
    

def plot():
    # Plot a performance comparison 
    # --------------------------------------------------------------------------------------------
    print("")
    num_samples = [10**i for i in range(4, 9)] 
    
    times_numpy = []; times_numba = []
    times_cupy = []; times_cupy_kernel = []
    times_cython = []
    
    num_runs = 5
    for n in num_samples:
        print(f"Estimating Pi using {n:,} samples.")
        time_numba = 0; time_numpy = 0; 
        time_cupy = 0; time_cupy_kernel = 0
        time_cython = 0        
        for _ in range(num_runs):           
            start = time.time()
            monteCarloPiNumpy(n)
            time_numpy += time.time() - start
            
            start = time.time()
            monteCarloPiNumba(n)
            time_numba += time.time() - start
            
            start = time.time()
            monteCarloPiCython(n)
            time_cython += time.time() - start

            if check_cupy():
                start = time.time()
                monteCarloPiCuPy(n)
                time_cupy += time.time() - start

                start = time.time()
                monteCarloPiCuPyKernel(n)
                time_cupy_kernel += time.time() - start

        times_numpy.append(time_numpy / num_runs)
        times_numba.append(time_numba / num_runs)
        times_cython.append(time_cython / num_runs)
        if check_cupy():
            times_cupy.append(time_cupy / num_runs)
            times_cupy_kernel.append(time_cupy_kernel / num_runs)
        
        print(f"\tAverage execution time for NumPy:       {times_numpy[-1]:.4f} seconds")
        print(f"\tAverage execution time for Numba:       {times_numba[-1]:.4f} seconds")
        print(f"\tAverage execution time for Cython:      {times_cython[-1]:.4f} seconds")
        if check_cupy():
            print(f"\tAverage execution time for CuPy:        {times_cupy[-1]:.4f} seconds")
            print(f"\tAverage execution time for CuPy Kernel: {times_cupy_kernel[-1]:.4f} seconds")
                
        
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(num_samples, times_numpy, label="NumPy")
    plt.plot(num_samples, times_numba, label="Numba")
    plt.plot(num_samples, times_cython, label="Cython")
    if check_cupy():
        plt.plot(num_samples, times_cupy, label="CuPy")
        plt.plot(num_samples, times_cupy_kernel, label="CuPy Kernel")
    # plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Number of Samples")
    plt.ylabel("Execution Time (s)")
    plt.title("Monte Carlo Pi Estimation Performance Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()
    # plt.savefig("misc/monte_carlo_pi.png")

        
if __name__ == "__main__":
    main()
    # plot()
    

# OUTPUT:
# Estimating Pi using 100,000,000 samples.
# Average execution time for monteCarloPiNumpy_AVG is 2.1847 seconds
# Average execution time for monteCarloPiNumba_AVG is 1.0806 seconds
# Average execution time for monteCarloPiCuPy_AVG is 0.4840 seconds
# Average execution time for monteCarloPiCuPyKernel_AVG is 0.0336 seconds
# 
# Pi estimate using NumPy:       3.1416502
# Pi estimate using Numba:       3.14157812
# Pi estimate using CuPy:        3.141483
# Pi estimate using CuPy Kernel: 3.14175208

# Estimating Pi using 10,000 samples.
#         Average execution time for NumPy:       0.0002 seconds
#         Average execution time for Numba:       0.1140 seconds
#         Average execution time for CuPy:        0.2349 seconds
#         Average execution time for CuPy Kernel: 0.0024 seconds
# Estimating Pi using 100,000 samples.
#         Average execution time for NumPy:       0.0022 seconds
#         Average execution time for Numba:       0.0010 seconds
#         Average execution time for CuPy:        0.0010 seconds
#         Average execution time for CuPy Kernel: 0.0004 seconds
# Estimating Pi using 1,000,000 samples.
#         Average execution time for NumPy:       0.0252 seconds
#         Average execution time for Numba:       0.0096 seconds
#         Average execution time for CuPy:        0.0044 seconds
#         Average execution time for CuPy Kernel: 0.0006 seconds
# Estimating Pi using 10,000,000 samples.
#         Average execution time for NumPy:       0.2303 seconds
#         Average execution time for Numba:       0.0948 seconds
#         Average execution time for CuPy:        0.0397 seconds
#         Average execution time for CuPy Kernel: 0.0036 seconds
# Estimating Pi using 100,000,000 samples.
#         Average execution time for NumPy:       2.1458 seconds
#         Average execution time for Numba:       0.9563 seconds
#         Average execution time for CuPy:        0.4791 seconds
#         Average execution time for CuPy Kernel: 0.0318 seconds