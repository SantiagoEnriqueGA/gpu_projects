import numpy as np
import cupy as cp
import pyopencl as cl
from numba import jit, cuda
from utils import *

# OpenCL context version, set to device
PYOPENCL_CTX_VERSION = '0'

@avg_timing_decorator
def fill_array(a, n):
    """Fill an array with sequential integers."""
    for i in range(n):
        a[i] = i
    return a


@avg_timing_decorator
@jit
def fill_array_numba(a, n):
    """Fill an array with sequential integers using Numba's @jit decorator."""
    for i in range(n):
        a[i] = i
    return a

@avg_timing_decorator
def fill_array_cupy(a, n):
    """Fill an array with sequential integers using CuPy."""
    a_cp = cp.asarray(a)
    for i in range(n):
        a_cp[i] = i
    return a_cp.get()
    
@avg_timing_decorator
def fill_array_opencl(a, n):
    """Fill an array with sequential integers using OpenCL."""
        # Create a program
    kernel_code = """
        __kernel void fill_array(__global float *a, const unsigned int n) {
            int i = get_global_id(0);
            if (i < n) {
                a[i] = i;
            }
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
    a_opencl = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, a.nbytes)
    
    # Execute the kernel
    program.fill_array(queue, a.shape, None, a_opencl, np.uint32(n))
    
    # Copy the result back to the host
    cl.enqueue_copy(queue, a, a_opencl).wait()   
    
    return a

def main():
    # Parameters
    N = 64_000_000

    print(f"Filling an array of {N:,} elements.")

    # Initialize an array of ones
    a = np.ones(N, dtype=np.float32)

    # Run the fill_array function
    a = fill_array(a, N)
    a_numba = fill_array_numba(a, N)
    # a_cupy = fill_array_cupy(a, N)
    a_opencl = fill_array_opencl(a, N)
    
    assert np.allclose(a, a_numba)
    # assert np.allclose(a, a_cupy)
    assert np.allclose(a, a_opencl)
    print(f"All elements of the result are equal!")
    
    
    
if __name__ == "__main__":
    main()    


# OUTPUT:
# Filling an array of 64,000,000 elements.
# Average execution time for fill_array is 7.3506 seconds
# Average execution time for fill_array_numba is 0.1083 seconds
# All elements of the result are equal!