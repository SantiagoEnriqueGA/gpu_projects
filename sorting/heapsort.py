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
    return np.sort(arr, kind="heapsort")

# -------------------------------------------------------------------------------------------------
# Functions to perform HeapSort
# -------------------------------------------------------------------------------------------------

@timing_decorator
def heapsort_cpu(arr):
    """Perform HeapSort on the CPU"""
    def _heapsort_cpu(arr):
        def heapify(arr, n, i):
            largest = i
            l = 2 * i + 1
            r = 2 * i + 2

            if l < n and arr[i] < arr[l]:
                largest = l

            if r < n and arr[largest] < arr[r]:
                largest = r

            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                heapify(arr, n, largest)

        n = len(arr)

        for i in range(n // 2 - 1, -1, -1):
            heapify(arr, n, i)

        for i in range(n-1, 0, -1):
            arr[i], arr[0] = arr[0], arr[i]
            heapify(arr, i, 0)

    _heapsort_cpu(arr)
    return arr


@timing_decorator
@jit(nopython=True)
def heapsort_numba(arr):
    """Perform HeapSort on the CPU using Numba Vectorized Functions."""
    _heapsort_numba(arr, len(arr))
    return arr

@jit(nopython=True)
def _heapsort_numba(arr, n):
    n = len(arr)

    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)

    return arr

@jit(nopython=True)
def heapify(arr, n, i):
        largest = i
        l = 2 * i + 1
        r = 2 * i + 2

        if l < n and arr[i] < arr[l]:
            largest = l

        if r < n and arr[largest] < arr[r]:
            largest = r

        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            heapify(arr, n, largest)


@timing_decorator
def heapsort_opencl(arr):
    """Perform HeapSort on the GPU using PyOpenCL"""
    # Set the environment variable
    os.environ['PYOPENCL_CTX'] = PYOPENCL_CTX_VERSION

    # OpenCL kernel code
    kernel_code = """
    __kernel void swap(__global int* arr, int i, int j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }

    __kernel void heapify(__global int* arr, int n, int i) {
        int largest = i; // Initialize largest as root Since we are using 0 based indexing
        int l = 2 * i + 1; // left = 2*i + 1
        int r = 2 * i + 2; // right = 2*i + 2

        // If left child is larger than root
        if (l < n && arr[l] > arr[largest])
            largest = l;

        // If right child is larger than largest so far
        if (r < n && arr[r] > arr[largest])
            largest = r;

        // If largest is not root
        if (largest != i) {
            int temp = arr[i];
            arr[i] = arr[largest];
            arr[largest] = temp;
        }
    }

    __kernel void heapsort(__global int* arr, int n) {
        // Build the heap (rearrange array)
        for (int i = n / 2 - 1; i >= 0; i--) {
            int k = i;
            while (k < n) {
                int largest = k;
                int l = 2 * k + 1;
                int r = 2 * k + 2;
                if (l < n && arr[l] > arr[largest]) largest = l;
                if (r < n && arr[r] > arr[largest]) largest = r;
                if (largest != k) {
                    int temp = arr[k];
                    arr[k] = arr[largest];
                    arr[largest] = temp;
                    k = largest;
                } else {
                    break;
                }
            }
        }

        // One by one extract an element from heap
        for (int i = n - 1; i >= 0; i--) {
            int temp = arr[0];
            arr[0] = arr[i];
            arr[i] = temp;
            int k = 0;
            while (k < i) {
                int largest = k;
                int l = 2 * k + 1;
                int r = 2 * k + 2;
                if (l < i && arr[l] > arr[largest]) largest = l;
                if (r < i && arr[r] > arr[largest]) largest = r;
                if (largest != k) {
                    int temp = arr[k];
                    arr[k] = arr[largest];
                    arr[largest] = temp;
                    k = largest;
                } else {
                    break;
                }
            }
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

    # Build and execute the kernel
    program = cl.Program(context, kernel_code).build()
    heapsort_kernel = program.heapsort
    heapsort_kernel.set_args(arr_buf, np.int32(len(arr)))

    cl.enqueue_nd_range_kernel(queue, heapsort_kernel, (1,), None)
    cl.enqueue_copy(queue, arr, arr_buf).wait()

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
    
    sorted_cpu = heapsort_cpu(arr.copy())
    
    sorted_numba = heapsort_numba(arr.copy())
    
    # Check for OpenCL-enabled GPU
    if check_openCl():
        sorted_openCl = heapsort_opencl(arr.copy())
    else:
        print("\n--OpenCL-enabled GPU not available!")
    

    assert np.allclose(sorted_numpy, sorted_cpu)
    assert np.allclose(sorted_numpy, sorted_numba)
    if check_openCl():
        assert np.allclose(sorted_numpy, sorted_openCl)
    print(f"All elements of the result are equal!")
    
    # # Plot a performance comparison 
    # # --------------------------------------------------------------------------------------------
    # arr_sizes = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]
    # cpu_times = []; numba_times = []; opencl_times = []; numpy_times = []
    # for size in arr_sizes:
    #     arr = np.random.randint(0, 1_000_000, size).astype(np.int32)
        
    #     print(f"\nSorting array of size {size:,} elements.")
        
    #     # CPU
    #     start = time.time()
    #     heapsort_cpu(arr.copy())
    #     cpu_times.append(time.time() - start)
        
    #     # NumPy
    #     start = time.time()
    #     sort_numpy(arr.copy())
    #     numpy_times.append(time.time() - start)
        
    #     # Numba
    #     start = time.time()
    #     heapsort_numba(arr.copy())
    #     numba_times.append(time.time() - start)
        
    #     # OpenCL
    #     if check_openCl():
    #         start = time.time()
    #         heapsort_opencl(arr.copy())
    #         opencl_times.append(time.time() - start)
    #     else:
    #         opencl_times.append(None)
            
    # # Plot the results
    # plt.figure(figsize=(10, 6))
    # plt.plot(arr_sizes, cpu_times, label="CPU")
    # plt.plot(arr_sizes, numba_times, label="Numba")
    # if opencl_times:
    #     plt.plot(arr_sizes, opencl_times, label="OpenCL")
    # plt.plot(arr_sizes, numpy_times, label="NumPy")
    # # plt.xscale("log")
    # plt.yscale("log")
    # plt.xlabel("Array size")
    # plt.ylabel("Time (s)")
    # plt.title("HeapSort Performance Comparison")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

if __name__ == "__main__":
    main()


# OUTPUT:
# Sorting array of size 1,000,000 elements.
# ------------------------------------------------------------
# Execution time for sort_numpy is 0.1383 seconds
# Execution time for heapsort_cpu is 11.7589 seconds
# Execution time for heapsort_numba is 1.3883 seconds
# Execution time for heapsort_opencl is 10.3213 seconds
# All elements of the result are equal!