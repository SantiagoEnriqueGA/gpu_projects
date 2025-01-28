import time
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import pyopencl as cl
import os

from utils import *

# OpenCL context version, set to device
PYOPENCL_CTX_VERSION = '0'

@timing_decorator
def sort_numpy(arr):
    """Sort an array using NumPy. This is the baseline implementation."""
    return np.sort(arr, kind="mergesort")

# -------------------------------------------------------------------------------------------------
# Functions to perform MergeSort
# -------------------------------------------------------------------------------------------------

@timing_decorator
def mergesort_cpu(arr):
    """Perform MergeSort on the CPU"""
    def _mergesort_cpu(arr):
        if len(arr) > 1:
            mid = len(arr) // 2
            left_half = np.copy(arr[:mid])
            right_half = np.copy(arr[mid:])

            _mergesort_cpu(left_half)
            _mergesort_cpu(right_half)

            i = j = k = 0

            while i < len(left_half) and j < len(right_half):
                if left_half[i] < right_half[j]:
                    arr[k] = left_half[i]
                    i += 1
                else:
                    arr[k] = right_half[j]
                    j += 1
                k += 1

            while i < len(left_half):
                arr[k] = left_half[i]
                i += 1
                k += 1

            while j < len(right_half):
                arr[k] = right_half[j]
                j += 1
                k += 1

    _mergesort_cpu(arr)
    return arr

@timing_decorator
@jit(nopython=True)
def mergesort_numba(arr):
    """Perform MergeSort on the CPU using Numba Vectorized Functions."""
    _mergesort_numba(arr)
    return arr

@jit(nopython=True)
def _mergesort_numba(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = np.copy(arr[:mid])
        right_half = np.copy(arr[mid:])

        _mergesort_numba(left_half)
        _mergesort_numba(right_half)

        i = j = k = 0

        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1

        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1


@timing_decorator
def mergesort_opencl(arr):
    """Perform MergeSort on the GPU using PyOpenCL"""
    # Set the environment variable
    os.environ['PYOPENCL_CTX'] = PYOPENCL_CTX_VERSION

    # OpenCL kernel code for merging two subarrays
    kernel_code = """
    __kernel void merge(__global int* arr, __global int* temp, const int width, const int n) {
        int gid = get_global_id(0);
        int left = gid * 2 * width;
        int right = min(left + width, n);
        int end = min(left + 2 * width, n);
        int l = left;
        int r = right;
        int t = left;

        while (l < right && r < end) {
            if (arr[l] < arr[r]) {
                temp[t++] = arr[l++];
            } else {
                temp[t++] = arr[r++];
            }
        }
        while (l < right) {
            temp[t++] = arr[l++];
        }
        while (r < end) {
            temp[t++] = arr[r++];
        }
        for (int i = left; i < end; i++) {
            arr[i] = temp[i];
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
    arr_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=arr)
    temp_buf = cl.Buffer(context, mf.READ_WRITE, arr.nbytes)

    # Build and execute the kernel
    program = cl.Program(context, kernel_code).build()
    merge_kernel = program.merge

    # Iteratively merge subarrays
    width = 1
    n = len(arr)
    while width < n:
        # Merge subarrays of size width
        merge_kernel.set_args(arr_buf, temp_buf, np.int32(width), np.int32(n))
        
        # Execute the kernel and copy the result back to the host
        cl.enqueue_nd_range_kernel(queue, merge_kernel, (n // (2 * width) + 1,), None)
        cl.enqueue_copy(queue, arr, arr_buf).wait()
        
        # Double the width
        width *= 2

    return arr


def main():
    # Parameters
    SIZE = 1_000_000
    
    arr = np.random.randint(0, 1_000_000, SIZE).astype(np.int32)
    
    print(f"Sorting array of size {SIZE:,} elements.")
    print("-"*60)
    
    # Runs single and shows the sorted array
    #--------------------------------------------------------------------------------------------
    sorted_numpy = sort_numpy(arr.copy())
    
    sorted_cpu = mergesort_cpu(arr.copy())
    
    sorted_numba = mergesort_numba(arr.copy())
    
    # Check for OpenCL-enabled GPU
    if check_openCl():
        sorted_openCl = mergesort_opencl(arr.copy())
    else:
        print("\n--OpenCL-enabled GPU not available!")


    assert np.allclose(sorted_numpy, sorted_cpu)
    assert np.allclose(sorted_numpy, sorted_numba)
    if check_openCl():
        assert np.allclose(sorted_numpy, sorted_openCl)
    print(f"All elements of the result are equal!")
    
    # Plot a performance comparison 
    # --------------------------------------------------------------------------------------------
    arr_sizes = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]
    cpu_times = []; numba_times = []; opencl_times = []; numpy_times = []
    for size in arr_sizes:
        arr = np.random.randint(0, 1_000_000, size).astype(np.int32)
        
        print(f"\nSorting array of size {size:,} elements.")
        
        # CPU
        start = time.time()
        cpu = mergesort_cpu(arr.copy())
        cpu_times.append(time.time() - start)
        
        # NumPy
        start = time.time()
        numpy = sort_numpy(arr.copy())
        numpy_times.append(time.time() - start)
        
        # Numba
        start = time.time()
        numba = mergesort_numba(arr.copy())
        numba_times.append(time.time() - start)
        
        # OpenCL
        if check_openCl():
            start = time.time()
            opencl = mergesort_opencl(arr.copy())
            opencl_times.append(time.time() - start)
        else:
            opencl_times.append(None)
            
        # Check if the results are equal
        assert np.allclose(cpu, numpy)
        assert np.allclose(cpu, numba)
        if check_openCl():
            assert np.allclose(cpu, opencl)
            
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(arr_sizes, cpu_times, label="CPU")
    plt.plot(arr_sizes, numba_times, label="Numba")
    if opencl_times:
        plt.plot(arr_sizes, opencl_times, label="OpenCL")
    plt.plot(arr_sizes, numpy_times, label="NumPy")
    # plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Array size")
    plt.ylabel("Time (s)")
    plt.title("MergeSort Performance Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()


# OUTPUT:
# Sorting array of size 1,000,000 elements.
# ------------------------------------------------------------
# Execution time for sort_numpy is 0.0631 seconds
# Execution time for mergesort_cpu is 8.0903 seconds
# Execution time for mergesort_numba is 1.2786 seconds
# Execution time for mergesort_opencl is 1.6243 seconds
# All elements of the result are equal!