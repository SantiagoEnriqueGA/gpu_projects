import time
import numpy as np
from numba import cuda
import matplotlib.pyplot as plt

from utils import *
from mandelbrot_funcs import *

# -------------------------------------------------------------------------------------------------
# Functions to run and time the Mandelbrot fractal on the CPU and GPU
# -------------------------------------------------------------------------------------------------
def run_cpu(size_multi=2, max_iters=50, show=True, save=False):
    """Run the Mandelbrot fractal on the CPU."""
    width = 750 * size_multi
    height = 500 * size_multi
    
    image = np.zeros((height, width), dtype=np.uint8)
    
    start_time = time.time()
    create_fractal_cpu(-2.0, 1.0, -1.0, 1.0, image, max_iters)
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"Execution time for create_fractal is {execution_time:.3f} seconds")
    
    if show or save:
        plt.imshow(image, cmap='seismic')
        plt.title(f"Mandelbrot Set (CPU, iters={max_iters})")
        plt.tight_layout()
    if show:
        plt.show()
    if save:
        plt.imsave(f"mandelbrot/plots/cpu_mandelbrot_size{size_multi}_iters{max_iters}.png", image, dpi=3000)
    return image

def run_gpu_cuda(size_multi=2, max_iters=50, show=True, save=False):
    """Run the Mandelbrot fractal on the GPU using Numba."""
    width = 750 * size_multi
    height = 500 * size_multi
    
    image = np.zeros((height, width), dtype=np.uint8)
    
    d_image = cuda.to_device(image)
    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(width / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(height / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    start_time = time.time()
    create_fractal_numba[blockspergrid, threadsperblock](-2.0, 1.0, -1.0, 1.0, d_image, max_iters)
    cuda.synchronize()
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"Execution time for create_fractal_numba is {execution_time:.4f} seconds")
    
    image = d_image.copy_to_host()

    if show or save:
        plt.imshow(image, cmap='seismic')
        plt.title(f"Mandelbrot Set (GPU, iters={max_iters})")
        plt.tight_layout()
    if show:
        plt.show()
    if save:
        plt.imsave(f"mandelbrot/plots/cuda_mandelbrot_size{size_multi}_iters{max_iters}.png", image, dpi=3000)
    return image


def run_gpu_opencl(size_multi=2, max_iters=50, show=True, save=False):
    """Run the Mandelbrot fractal on the GPU using PyOpenCL."""
    width = 750 * size_multi
    height = 500 * size_multi
    
    image = np.zeros((height, width), dtype=np.uint8)
    
    start_time = time.time()
    create_fractal_opencl(-2.0, 1.0, -1.0, 1.0, image, max_iters)
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"Execution time for create_fractal_opencl is {execution_time:.4f} seconds")
    
    if show or save:
        plt.imshow(image, cmap='seismic')
        plt.title(f"Mandelbrot Set (GPU with PyOpenCL, iters={max_iters})")
        plt.tight_layout()
    if show:
        plt.show()
    if save:
        plt.imsave(f"mandelbrot/plots/opencl_mandelbrot_size{size_multi}_iters{max_iters}.png", image, dpi=3000)
    return image


def run_gpu_cupy(size_multi, max_iters, show=True, save=False):
    width = 750 * size_multi
    height = 500 * size_multi
    
    image = np.zeros((height, width), dtype=np.uint8)
    
    start_time = time.time()
    create_fractal_cupy(-2.0, 1.0, -1.0, 1.0, image, max_iters)
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Execution time for {max_iters} iterations is {execution_time:.4f} seconds")
    
    if show or save:
        plt.imshow(image, cmap='seismic')
        plt.title(f"Mandelbrot Set (GPU with CuPy, iters={max_iters})")
        plt.tight_layout()
    if show:
        plt.show()
    if save:
        plt.imsave(f"mandelbrot/plots/cupy_mandelbrot_size{size_multi}_iters{max_iters}.png", image, dpi=3000)
    return image


def main():
    # Parameters
    SIZE_MULTI = 2
    MAX_ITERS = 50
    SHOW_IMAGE = True
    SAVE_IMAGE = True
    CPU = True
    
    print(f"Creating Mandelbrot Set of size {750*SIZE_MULTI}x{500*SIZE_MULTI} with {MAX_ITERS} iterations")
    print(f"\tNumber of pixels:    {750*SIZE_MULTI*500*SIZE_MULTI:,}")
    print(f"\tPixels x Iterations: {750*SIZE_MULTI*500*SIZE_MULTI*MAX_ITERS:,}")
    print("-"*60)
    
    # Runs single and shows the Mandelbrot fractal 
    #--------------------------------------------------------------------------------------------
    if CPU:
        print("\nRunning CPU version...")
        run_cpu(size_multi=SIZE_MULTI, max_iters=MAX_ITERS, show=SHOW_IMAGE, save=SAVE_IMAGE)
        
    # Check for CUDA-enabled GPU
    if check_numba_cuda():
        print("\nRunning GPU Numba CUDA version...")    # Requires a CUDA-enabled GPU
        run_gpu_cuda(size_multi=SIZE_MULTI, max_iters=MAX_ITERS, show=SHOW_IMAGE, save=SAVE_IMAGE)
    else:
        print("\n--CUDA-enabled GPU not available!")
    
    # Check for OpenCL-enabled GPU
    if check_openCl():
        print("\nRunning GPU OpenCL version...")  # Requires an OpenCL-enabled GPU
        run_gpu_opencl(size_multi=SIZE_MULTI, max_iters=MAX_ITERS, show=SHOW_IMAGE, save=SAVE_IMAGE)
    else:
        print("\n--OpenCL-enabled GPU not available!")
    
    # Check for CuPy-enabled GPU
    if check_cupy():
        print("\nRunning GPU CuPy version...")  # Requires a CuPy-enabled GPU
        run_gpu_cupy(size_multi=SIZE_MULTI, max_iters=MAX_ITERS, show=SHOW_IMAGE, save=SAVE_IMAGE)
    else:
        print("\n--CuPy-enabled GPU not available!")

if __name__ == "__main__":
    main()
