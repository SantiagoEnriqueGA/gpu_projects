import time
import numpy as np
from numba import cuda

from utils import *
from julia_funcs import *

# -------------------------------------------------------------------------------------------------
# Functions to run and time the Julia Set fractal on the CPU and GPU
# -------------------------------------------------------------------------------------------------
@avg_timing_decorator
def run_cpu_avg(size_multi=2, max_iters=50):
    """Run the Julia Set fractal on the CPU and calculate the average execution time."""
    width = 750 * size_multi
    height = 500 * size_multi
    image = np.zeros((height, width), dtype=np.uint8)
    create_fractal_cpu(-2.0, 1.0, -1.0, 1.0, image, max_iters)


@avg_timing_decorator
def run_gpu_cuda_avg(size_multi=2, max_iters=50):
    """Run the Julia Set fractal on the GPU using Numba and calculate the average execution time."""
    width = 750 * size_multi
    height = 500 * size_multi
    
    image = np.zeros((height, width), dtype=np.uint8)
    d_image = cuda.to_device(image)
    
    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(width / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(height / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    create_fractal_numba[blockspergrid, threadsperblock](-2.0, 1.0, -1.0, 1.0, d_image, max_iters)
    cuda.synchronize()
    image = d_image.copy_to_host()


@avg_timing_decorator
def run_gpu_opencl_avg(size_multi=2, max_iters=50):
    """Run the Julia Set fractal on the GPU using PyOpenCL and calculate the average execution time."""
    width = 750 * size_multi
    height = 500 * size_multi
    image = np.zeros((height, width), dtype=np.uint8)
    create_fractal_opencl(-2.0, 1.0, -1.0, 1.0, image, max_iters)

@avg_timing_decorator
def run_gpu_cupy_avg(size_multi=2, max_iters=50):
    """Run the Julia Set fractal on the GPU using CuPy and calculate the average execution time."""
    width = 750 * size_multi
    height = 500 * size_multi
    image = np.zeros((height, width), dtype=np.uint8)
    create_fractal_cupy(-2.0, 1.0, -1.0, 1.0, image, max_iters)

def main():
    # Parameters
    SIZE_MULTI = 2
    MAX_ITERS = 50
    
    print(f"Creating Julia Set Set of size {750*SIZE_MULTI}x{500*SIZE_MULTI} with {MAX_ITERS} iterations")
    print(f"\tNumber of pixels:    {750*SIZE_MULTI*500*SIZE_MULTI:,}")
    print(f"\tPixels x Iterations: {750*SIZE_MULTI*500*SIZE_MULTI*MAX_ITERS:,}")
    print("-"*60)
    
    
    # Runs 5 times and calculates the average time taken to run the Julia Set fractal
    #--------------------------------------------------------------------------------------------
    print("Running CPU version (average timing)...")
    run_cpu_avg(size_multi=SIZE_MULTI, max_iters=MAX_ITERS)

    # Check for CUDA-enabled GPU
    if check_numba_cuda():
        print("\nRunning GPU CUDA version (average timing)...")   # Requires a CUDA-enabled GPU
        run_gpu_cuda_avg(size_multi=SIZE_MULTI, max_iters=MAX_ITERS)
    else:
        print("\n--CUDA-enabled GPU not available!")
    
    # Check for OpenCL-enabled GPU
    if check_openCl():
        print("\nRunning GPU OpenCL version (average timing)...") # Requires an OpenCL-enabled GPU
        run_gpu_opencl_avg(size_multi=SIZE_MULTI, max_iters=MAX_ITERS)
    else:
        print("\n--OpenCl-enabled GPU not available!")

    # Check for CuPy-enabled GPU
    if check_cupy():
        print("\nRunning GPU CuPy version (average timing)...") # Requires a CuPy-enabled GPU
        run_gpu_cupy_avg(size_multi=SIZE_MULTI, max_iters=MAX_ITERS)
    else:
        print("\n--CuPy-enabled GPU not available!")


if __name__ == "__main__":
    main()
