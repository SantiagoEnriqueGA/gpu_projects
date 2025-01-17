import time
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import pyopencl as cl
import os

from utils import timing_decorator, avg_timing_decorator
from utils import suppress_output, enable_output
from utils import check_numba_cuda, check_openCl

# OpenCL context version, set to device
PYOPENCL_CTX_VERSION = '1'

# -------------------------------------------------------------------------------------------------
# Functions to perform QuickSort
# -------------------------------------------------------------------------------------------------
@timing_decorator
def sort_numpy(arr):
    return np.sort(arr)

@timing_decorator
def quicksort_cpu(arr):
    """Perform QuickSort on the CPU"""
    def _quicksort_cpu(arr):
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return _quicksort_cpu(left) + middle + _quicksort_cpu(right)
    return _quicksort_cpu(arr)


@timing_decorator
@jit(nopython=True)
def quicksort_numba(arr):
    _quicksort_numba(arr, 0, len(arr) - 1)
    return list(arr)

@jit(nopython=True)
def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

@jit(nopython=True)
def _quicksort_numba(arr, low, high):
    if low < high:
        pi = partition(arr, low, high)
        _quicksort_numba(arr, low, pi - 1)
        _quicksort_numba(arr, pi + 1, high)


@timing_decorator
def quicksort_opencl(arr):
    """Perform QuickSort on the GPU using PyOpenCL"""
    # Set the environment variable
    os.environ['PYOPENCL_CTX'] = PYOPENCL_CTX_VERSION
    
    # OpenCL kernel code
    kernel_code = """
    __kernel void quicksort(__global int* arr, int left, int right) {
        int i = left;
        int j = right;
        int pivot = arr[(left + right) / 2];
        int temp;
        
        while (i <= j) {
            while (arr[i] < pivot) i++;
            while (arr[j] > pivot) j--;
            if (i <= j) {
                temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
                i++;
                j--;
            }
        }
        
        if (left < j) quicksort(arr, left, j);
        if (i < right) quicksort(arr, i, right);
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
    
    # Build and execute the kernel
    program = cl.Program(context, kernel_code).build()
    quicksort_kernel = program.quicksort
    quicksort_kernel.set_args(arr_buf, np.int32(0), np.int32(len(arr) - 1))
    
    cl.enqueue_nd_range_kernel(queue, quicksort_kernel, (1,), None)
    cl.enqueue_copy(queue, arr, arr_buf).wait()
    
    return list(arr)


def main():
    # Parameters
    SIZE = 1_000_000
    
    arr = np.random.randint(0, 1_000_000, SIZE).astype(np.int32)
    
    print(f"Sorting array of size {SIZE:,} elements.")
    print("-"*60)
    
    # Runs single and shows the sorted array
    #--------------------------------------------------------------------------------------------
    print("\nRunning CPU version...")
    sorted_cpu = quicksort_cpu(arr.copy())
    print(f"First 10 elements CPU: {sorted_cpu[:10]}")
    
    print("\nRunning NumPy version...")
    sorted_numpy = sort_numpy(arr.copy())
    print(f"First 10 elements NumPy: {sorted_numpy[:10]}")
    
    print("\nRunning Numba version...")
    sorted_numba = quicksort_numba(arr.copy())
    print(f"First 10 elements Numba: {sorted_numba[:10]}")
    
    # Check for OpenCL-enabled GPU
    if check_openCl():
        print("\nRunning GPU OpenCL version...")
        sorted_openCl = quicksort_opencl(arr.copy())
        print(f"First 10 elements OpenCL: {sorted_openCl[:10]}")
    else:
        print("\n--OpenCL-enabled GPU not available!")
    
    
    # Plot a performance comparison 
    # --------------------------------------------------------------------------------------------
    arr_sizes = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]
    cpu_times = []; numba_times = []; opencl_times = []; numpy_times = []
    for size in arr_sizes:
        arr = np.random.randint(0, 1_000_000, size).astype(np.int32)
        
        print(f"\nSorting array of size {size:,} elements.")
        
        # CPU
        start = time.time()
        quicksort_cpu(arr.copy())
        cpu_times.append(time.time() - start)
        
        # NumPy
        start = time.time()
        sort_numpy(arr.copy())
        numpy_times.append(time.time() - start)
        
        # Numba
        start = time.time()
        quicksort_numba(arr.copy())
        numba_times.append(time.time() - start)
        
        # OpenCL
        if check_openCl():
            start = time.time()
            quicksort_opencl(arr.copy())
            opencl_times.append(time.time() - start)
        else:
            opencl_times.append(None)
            
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
    plt.title("QuickSort Performance Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()
    

if __name__ == "__main__":
    main()
