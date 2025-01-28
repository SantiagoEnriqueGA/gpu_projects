import time
import numpy as np
import cupy as cp
from numba import jit
import matplotlib.pyplot as plt
import pyopencl as cl
import os

from utils import *

# OpenCL context version, set to device
PYOPENCL_CTX_VERSION = '0'

def primes_cpu(max_time):
    """Generate prime numbers within a specified period of time using CPU"""
    start_time = time.time()
    primes = []
    num = 2
    while time.time() - start_time < max_time:
        is_prime = True
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
        num += 1
    return primes

def primes_numba(max_time):
    """Generate prime numbers within a specified period of time using Numba"""
    start_time = time.time()
    primes = []
    for prime in _primes_numba_helper():
        if time.time() - start_time >= max_time:
            break
        primes.append(prime)
    return primes

@jit(nopython=True)
def _primes_numba_helper():
    """Helper function to generate prime numbers using Numba"""
    num = 2
    while True:
        is_prime = True
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            yield num
        num += 1

def primes_opencl(max_time, batch_size=1000):
    """Generate prime numbers within a specified period of time using PyOpenCL"""
    start_time = time.time()
    
    # OpenCL kernel to check if numbers are prime
    kernel_code = """
    __kernel void is_prime(__global const int *numbers, __global int *results) {
        int idx = get_global_id(0);
        int num = numbers[idx];
        int is_prime = 1;
        if (num < 2) {
            is_prime = 0;
        } else {
            for (int i = 2; i <= sqrt((float)num); ++i) {
                if (num % i == 0) {
                    is_prime = 0;
                    break;
                }
            }
        }
        results[idx] = is_prime;
    }
    """
    # Initialize OpenCL
    platform = cl.get_platforms()[int(PYOPENCL_CTX_VERSION)]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)
    
    # Compile the kernel
    program = cl.Program(context, kernel_code).build()
    
    # Initialize variables
    primes = []
    num = 2
    
    while time.time() - start_time < max_time:
        # Create number batch to check
        numbers = np.arange(num, num + batch_size).astype(np.int32)
        results = np.zeros_like(numbers).astype(np.int32)
        
        # Create buffers
        numbers_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=numbers)
        results_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, results.nbytes)
        
        # Execute kernel
        program.is_prime(queue, numbers.shape, None, numbers_buf, results_buf)
        
        # Retrieve results
        cl.enqueue_copy(queue, results, results_buf).wait()
        
        # Add primes to the list
        primes.extend(numbers[results == 1])
        
        # Update num for next batch
        num += batch_size
        
        # Check time limit
        if time.time() - start_time >= max_time:
            break
        
    return primes

def primes_cupy(max_time, batch_size=1000):
    """Generate prime numbers within a specified period of time using CuPy"""
    start_time = time.time()
    
    # CuPy kernel to check if numbers are prime
    kernel_code = '''
    extern "C" __global__
    void is_prime(const int *numbers, int *results, int size) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < size) {
            int num = numbers[idx];
            int is_prime = 1;
            if (num < 2) {
                is_prime = 0;
            } else {
                for (int i = 2; i <= sqrtf((float)num); ++i) {
                    if (num % i == 0) {
                        is_prime = 0;
                        break;
                    }
                }
            }
            results[idx] = is_prime;
        }
    }
    '''
    
    # Compile the kernel
    is_prime_kernel = cp.RawKernel(kernel_code, 'is_prime')
    
    # Initialize variables
    primes = []
    num = 2
    
    while time.time() - start_time < max_time:
        # Create number batch to check
        numbers = cp.arange(num, num + batch_size, dtype=cp.int32)
        results = cp.zeros_like(numbers, dtype=cp.int32)
        
        # Execute kernel
        threads_per_block = 256
        blocks_per_grid = (batch_size + (threads_per_block - 1)) // threads_per_block
        is_prime_kernel((blocks_per_grid,), (threads_per_block,), (numbers, results, batch_size))
        
        # Retrieve results
        cp.cuda.stream.get_current_stream().synchronize()
        
        # Check time limit
        if time.time() - start_time >= max_time:
            break
        
        # Add primes to the list
        primes.extend(cp.asnumpy(numbers[results == 1]))
        
        # Update num for next batch
        num += batch_size
        
        
    return primes

def main():
    # Parameters
    MAX_TIME = 30
    
    print(f"Generating prime numbers within {MAX_TIME} seconds.")
    print("-"*60)
    
    # Generate primes using different methods
    primes_cpu_result = primes_cpu(MAX_TIME)
    print(f"CPU    produced {len(primes_cpu_result):<10,} prime numbers in {MAX_TIME} seconds")
    
    primes_numba_result = primes_numba(MAX_TIME)
    print(f"Numba  produced {len(primes_numba_result):<10,} prime numbers in {MAX_TIME} seconds")
    
    if check_openCl():
        primes_opencl_result = primes_opencl(MAX_TIME, 1000000)
        print(f"OpenCL produced {len(primes_opencl_result):<10,} prime numbers in {MAX_TIME} seconds")
    else: primes_opencl_result = None
        
    if check_cupy():
        primes_cupy_result = primes_cupy(MAX_TIME, 1000000)
        print(f"CuPy   produced {len(primes_cupy_result):<10,} prime numbers in {MAX_TIME} seconds")
    else: primes_cupy_result = None
    
    # Check results (limited to smallest result)
    smallest = min(len(primes_cpu_result), 
                   len(primes_numba_result), 
                   len(primes_opencl_result) if primes_opencl_result else float('inf'), 
                   len(primes_cupy_result) if primes_cupy_result else float('inf')) 
    assert np.array_equal(primes_cpu_result[:smallest], primes_numba_result[:smallest])
    if primes_opencl_result:
        assert np.array_equal(primes_cpu_result[:smallest], primes_opencl_result[:smallest])
    if primes_cupy_result:
        assert np.array_equal(primes_cpu_result[:smallest], primes_cupy_result[:smallest])
    print(f"All methods produced the same results!")    
    
def plot():
    time_lims = [1, 5, 10, 15, 20, 25, 30]
    cpu_cnts = []; numba_cnts = []; opencl_cnts = []; cupy_cnts = []
    for time_lim in time_lims:
        print(f"Generating prime numbers within {time_lim} seconds.")
        
        # CPU
        cpu_cnts.append(len(primes_cpu(time_lim)))
        
        # Numba
        numba_cnts.append(len(primes_numba(time_lim)))
        
        # OpenCL
        if check_openCl():
            opencl_cnts.append(len(primes_opencl(time_lim, 1000000)))
        
        # CuPy
        if check_cupy():
            cupy_cnts.append(len(primes_cupy(time_lim, 1000000)))
                             
    plt.figure(figsize=(10, 6))
    plt.plot(time_lims, cpu_cnts, label="CPU")
    plt.plot(time_lims, numba_cnts, label="Numba")
    if opencl_cnts:
        plt.plot(time_lims, opencl_cnts, label="OpenCL")
    if cupy_cnts:
        plt.plot(time_lims, cupy_cnts, label="CuPy")
    plt.yscale("log")
    plt.xlabel("Time Limit")
    plt.ylabel("Prime Count")
    plt.title("Prime Number Generation, to Time Limit, Performance Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()
        
        
        
    
if __name__ == "__main__":
    # main()
    plot()
    


# OUTPUT
# Generating prime numbers within 30 seconds.
# ------------------------------------------------------------
# CPU    produced 317,130    prime numbers in 30 seconds
# Numba  produced 2,864,911  prime numbers in 30 seconds
# OpenCL produced 5,653,023  prime numbers in 30 seconds
# All methods produced the same results!