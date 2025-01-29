import time
import numpy as np
import cupy as cp
from numba import jit
import matplotlib.pyplot as plt
import pyopencl as cl
import os

from utils import *

# OpenCL context version, set to device
PYOPENCL_CTX_VERSION = '1'

@timing_decorator
def primes_cpu(limit):
    """Generate prime numbers up to a given limit using CPU, Sieve of Eratosthenes"""
    prime = [True for i in range(limit + 1)]
    p = 2
    while p**2 <= limit:
        if prime[p] == True:
            for i in range(p**2, limit + 1, p):
                prime[i] = False
        p += 1
    primes = []
    for p in range(2, limit + 1):
        if prime[p]:
            primes.append(p)
    return primes
    
@timing_decorator
@jit(nopython=True)
def primes_numba(limit):
    """Generate prime numbers up to a given limit using Numba"""
    prime = [True for i in range(limit + 1)]
    p = 2
    while p**2 <= limit:
        if prime[p] == True:
            for i in range(p**2, limit + 1, p):
                prime[i] = False
        p += 1
    primes = []
    for p in range(2, limit + 1):
        if prime[p]:
            primes.append(p)
    return primes

@timing_decorator
def primes_opencl(limit):
    """Generate prime numbers up to a given limit using PyOpenCL"""
    # Set the environment variable
    os.environ['PYOPENCL_CTX'] = PYOPENCL_CTX_VERSION

    # OpenCL kernel code
    kernel_code = """
    __kernel void primes(__global int* primes, int limit) {
        int gid = get_global_id(0);
        if (gid < 2 || gid > limit) return;
        int is_prime = 1;
        for (int i = 2; i <= sqrt((float)gid); i++) {
            if (gid % i == 0) {
                is_prime = 0;
                break;
            }
        }
        if (is_prime) {
            primes[gid] = gid;
        } else {
            primes[gid] = 0;
        }
    }
    """

    # Initialize OpenCL
    platform = cl.get_platforms()[int(PYOPENCL_CTX_VERSION)]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)

    # Create buffers
    mf = cl.mem_flags
    primes = np.zeros(limit + 1, dtype=np.int32)
    primes_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=primes)

    # Build and execute the kernel
    program = cl.Program(context, kernel_code).build()
    primes_kernel = program.primes
    primes_kernel.set_args(primes_buf, np.int32(limit))

    cl.enqueue_nd_range_kernel(queue, primes_kernel, (limit + 1,), None)
    cl.enqueue_copy(queue, primes, primes_buf).wait()

    return primes[primes > 0]

@timing_decorator
def primes_cupy(limit):
    """Generate prime numbers up to a given limit using CuPy with the Sieve of Eratosthenes."""
    primes = cp.ones(limit + 1, dtype=cp.bool_)  # Initialize all numbers as prime
    primes[:2] = 0  # Mark 0 and 1 as non-prime

    for num in range(2, int(cp.sqrt(limit)) + 1):
        if primes[num]:  # If num is still marked prime
            primes[num * num : limit + 1 : num] = 0  # Mark multiples as non-prime

    return cp.asnumpy(cp.nonzero(primes)[0])  # Extract prime numbers


def main():
    # Parameters
    LIMIT = 100_000_00
    
    print(f"Generating prime numbers up to {LIMIT:,}.")
    print("-"*60)
    
    # Generate primes using different methods
    primes_cpu_result = primes_cpu(LIMIT)
    primes_numba_result = primes_numba(LIMIT)
    if check_openCl():
        primes_opencl_result = primes_opencl(LIMIT)
    if check_cupy():
        primes_cupy_result = primes_cupy(LIMIT)
    
    # Verify results
    assert np.array_equal(primes_cpu_result, primes_numba_result)
    if check_openCl():
        assert np.array_equal(primes_cpu_result, primes_opencl_result)
    if check_cupy():
        assert np.array_equal(primes_cpu_result, primes_cupy_result)
    print(f"All methods produced the same results!")
    
    # Print the last 5 prime numbers
    print(f"Last 5 prime numbers: {primes_cpu_result[-5:]}")

def plot():
    limits = [10 ** i for i in range(1, 9)]
    cpu_times = []; numba_times = []; opencl_times = []; cupy_times = []
    for limit in limits:
        print(f"Generating prime numbers up to {limit:,}.")
        
        # CPU
        start = time.time()
        primes_cpu(limit)
        cpu_times.append(time.time() - start)
        
        # Numba
        start = time.time()
        primes_numba(limit)
        numba_times.append(time.time() - start)
        
        # OpenCL
        if check_openCl():
            start = time.time()
            primes_opencl(limit)
            opencl_times.append(time.time() - start)
        
        # CuPy
        if check_cupy():
            start = time.time()
            primes_cupy(limit)
            cupy_times.append(time.time() - start)
            
    plt.figure(figsize=(10, 6))
    plt.plot(limits, cpu_times, label="CPU")
    plt.plot(limits, numba_times, label="Numba")
    if opencl_times:
        plt.plot(limits, opencl_times, label="OpenCL")
    if cupy_times:
        plt.plot(limits, cupy_times, label="CuPy")
    plt.yscale("log")
    plt.xlabel("Limit")
    plt.ylabel("Execution time (s)")
    plt.title("Prime Number Generation, Limited Time, Performance Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
    # plot()


# OUTPUT
# Generating prime numbers up to 10,000,000.
# ------------------------------------------------------------
# Execution time for primes_cpu is 1.2996 seconds
# Execution time for primes_numba is 0.6365 seconds
# Execution time for primes_opencl is 1.2198 seconds
# All methods produced the same results!
# Last 5 prime numbers: [9999937, 9999943, 9999971, 9999973, 9999991]


# ------------------------------------------------------------
# Archive: Old versions
# ------------------------------------------------------------
# @timing_decorator
# def primes_cpu(limit):
#     """Generate prime numbers up to a given limit using CPU"""
#     primes = []
#     for num in range(2, limit + 1):
#         is_prime = True
#         for i in range(2, int(num**0.5) + 1):
#             if num % i == 0:
#                 is_prime = False
#                 break
#         if is_prime:
#             primes.append(num)
#     return primes


# @timing_decorator
# @jit(nopython=True)
# def primes_numba(limit):
#     """Generate prime numbers up to a given limit using Numba"""
#     primes = []
#     for num in range(2, limit + 1):
#         is_prime = True
#         for i in range(2, int(num**0.5) + 1):
#             if num % i == 0:
#                 is_prime = False
#                 break
#         if is_prime:
#             primes.append(num)
#     return primes


# @timing_decorator
# def primes_cupy(limit):
#     """Generate prime numbers up to a given limit using CuPy (GPU acceleration)."""
#     arr = cp.arange(limit + 1)

#     # Start with all numbers marked as prime (1 = prime, 0 = not prime)
#     primes = cp.ones(limit + 1, dtype=cp.bool_)
#     primes[:2] = 0  # Mark 0 and 1 as non-prime

#     for num in range(2, int(cp.sqrt(limit)) + 1):
#         if primes[num]:  # If num is still marked prime
#             primes[num * num : limit + 1 : num] = 0  # Mark multiples as non-prime

#     return cp.asnumpy(arr[primes])  # Return only the prime numbers