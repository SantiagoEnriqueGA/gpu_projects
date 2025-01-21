import numpy as np
from numba import jit, config
import matplotlib.pyplot as plt
import time

from utils import *

@avg_timing_decorator
def monteCarloPiNumpy_AVG(num_samples):
    """Estimate Pi using Monte Carlo method with NumPy."""
    return monteCarloPiNumpy(num_samples)

@avg_timing_decorator
def monteCarloPiNumba_AVG(num_samples):
    """Estimate Pi using Monte Carlo method with Numba JIT."""
    return monteCarloPiNumpy(num_samples)


def monteCarloPiNumpy(num_samples):
    """Estimate Pi using Monte Carlo method with NumPy."""
    x = np.random.rand(num_samples)
    y = np.random.rand(num_samples)
    inside_circle = (x**2 + y**2) <= 1.0
    pi_estimate = (inside_circle.sum() / num_samples) * 4
    return pi_estimate

# The keyword argument 'parallel=True' was specified but no transformation for parallel execution was possible.
# @jit(nopython=True, parallel=True)
@jit(nopython=True)
def monteCarloPiNumba(num_samples):
    """Estimate Pi using Monte Carlo method with Numba JIT."""
    inside_circle = 0
    for _ in range(num_samples):
        x = np.random.rand()
        y = np.random.rand()
        if x**2 + y**2 <= 1.0:
            inside_circle += 1
    pi_estimate = (inside_circle / num_samples) * 4
    return pi_estimate

def main():
    # Parameters
    num_samples = 100_000_000
    
    print(f"Estimating Pi using {num_samples:,} samples.")

    pi_numpy = monteCarloPiNumpy_AVG(num_samples)
    pi_numba = monteCarloPiNumba_AVG(num_samples)

    print(f"Pi estimate using NumPy: {pi_numpy}")
    print(f"Pi estimate using Numba: {pi_numba}")
    
    # Plot a performance comparison 
    # --------------------------------------------------------------------------------------------
    print("")
    num_samples = [10**i for i in range(4, 9)]
    times_numpy = []; times_numba = []
    num_runs = 5
    for n in num_samples:
        print(f"Estimating Pi using {n:,} samples.")
        time_numba = 0; time_numpy = 0
        for _ in range(num_runs):
            start = time.time()
            monteCarloPiNumpy(n)
            time_numpy += time.time() - start
            
            start = time.time()
            monteCarloPiNumba(n)
            time_numba += time.time() - start
        times_numpy.append(time_numpy / num_runs)
        times_numba.append(time_numba / num_runs)
        
        print(f"\tAverage execution time for NumPy: {times_numpy[-1]:.4f} seconds")
        print(f"\tAverage execution time for Numba: {times_numba[-1]:.4f} seconds")
                
        
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(num_samples, times_numpy, label="NumPy")
    plt.plot(num_samples, times_numba, label="Numba")
    plt.xscale('log')
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
    

# OUTPUT:
# Estimating Pi using 100,000,000 samples.
# Average execution time for monteCarloPiNumpy_AVG is 2.1093 seconds
# Average execution time for monteCarloPiNumba_AVG is 1.9931 seconds
# Pi estimate using NumPy: 3.14150216
# Pi estimate using Numba: 3.14172632

# Estimating Pi using 10,000 samples.
#         Average execution time for NumPy: 0.0002 seconds
#         Average execution time for Numba: 0.1284 seconds
# Estimating Pi using 100,000 samples.
#         Average execution time for NumPy: 0.0029 seconds
#         Average execution time for Numba: 0.0011 seconds
# Estimating Pi using 1,000,000 samples.
#         Average execution time for NumPy: 0.0205 seconds
#         Average execution time for Numba: 0.0089 seconds
# Estimating Pi using 10,000,000 samples.
#         Average execution time for NumPy: 0.2071 seconds
#         Average execution time for Numba: 0.0950 seconds
# Estimating Pi using 100,000,000 samples.
#         Average execution time for NumPy: 2.0562 seconds
#         Average execution time for Numba: 1.0004 seconds