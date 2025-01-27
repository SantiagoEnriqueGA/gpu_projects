import numpy as np
from numba import cuda
import pyopencl as cl
import cupy as cp
import os

from utils import timing_decorator, avg_timing_decorator
from utils import suppress_output, enable_output
from utils import check_numba_cuda, check_openCl

PYOPENCL_CTX_VERSION = '1'

# -------------------------------------------------------------------------------------------------
# Functions to create the Mandelbrot fractal on the CPU and GPU
# -------------------------------------------------------------------------------------------------
def create_fractal_cpu(min_x, max_x, min_y, max_y, image, iters):
    """Create a Julia fractal on the CPU"""

    def julia(c, max_iter):
        """Calculate the Julia value for a given coordinate"""
        z = c
        for i in range(max_iter):
            z = z * z + complex(-0.7, 0.27015)
            if (z.real * z.real + z.imag * z.imag) >= 4:
                return i
        return 255

    # Set the width and height of the image
    width = image.shape[1]
    height = image.shape[0]

    # Set the pixel size in the x and y directions
    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    # For each pixel in the image, calculate the Julia value
    for x in range(width):
        real = min_x + x * pixel_size_x
        for y in range(height):
            imag = min_y + y * pixel_size_y
            color = julia(complex(real, imag), iters)
            image[y, x] = color

@cuda.jit
def create_fractal_numba(min_x, max_x, min_y, max_y, image, iters):
    """Create a Julia fractal on the GPU using Numba"""
    
    # Set the width and height of the image
    width = image.shape[1]
    height = image.shape[0]

    # Set the pixel size in the x and y directions
    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    # Get the x and y coordinates of the current thread
    x, y = cuda.grid(2)

    # If the x and y coordinates are within the image bounds
    if x < width and y < height:
        # Get the real and imaginary parts of the complex number
        real = min_x + x * pixel_size_x
        imag = min_y + y * pixel_size_y
        
        # Convert the x, y coordinates to a complex number
        z = complex(real, imag)
        c = complex(-0.7, 0.27015)
        
        # For each iteration, calculate z = z^2 + c
        for i in range(iters):
            z = z * z + c
            if (z.real * z.real + z.imag * z.imag) >= 4:
                image[y, x] = i     # Set the pixel value to the number of iterations
                return
        image[y, x] = 255           # If the magnitude of z never exceeds, set the pixel value to 255 (white)

def create_fractal_opencl(min_x, max_x, min_y, max_y, image, iters):
    """Create a Julia fractal on the GPU using PyOpenCL"""
    # Set the environment variable
    os.environ['PYOPENCL_CTX'] = PYOPENCL_CTX_VERSION
    
    # OpenCL kernel code
    # Similar to the CUDA kernel code, written in OpenCL C
    kernel_code = r"""
    __kernel void julia(
        const double min_x, const double max_x, const double min_y, const double max_y,
        __global uchar *image, const int width, const int height, const int iters) {
        
        int x = get_global_id(0);
        int y = get_global_id(1);
        
        if (x < width && y < height) {
            double pixel_size_x = (max_x - min_x) / width;
            double pixel_size_y = (max_y - min_y) / height;
            double real = min_x + x * pixel_size_x;
            double imag = min_y + y * pixel_size_y;
            double z_real = real;
            double z_imag = imag;
            double c_real = -0.7;
            double c_imag = 0.27015;
            int i;
            for (i = 0; i < iters; i++) {
                double z_real2 = z_real * z_real - z_imag * z_imag + c_real;
                double z_imag2 = 2.0 * z_real * z_imag + c_imag;
                z_real = z_real2;
                z_imag = z_imag2;
                if (z_real * z_real + z_imag * z_imag >= 4.0) {
                    break;
                }
            }
            image[y * width + x] = (uchar)(255.0 * i / iters);

        }
    }
    """
    
    suppress_output() # Suppress the output from PyOpenCL (optional)
    
    # Setup OpenCL
    context = cl.create_some_context()  # Create a context, i.e. the device to run the code
    queue = cl.CommandQueue(context)    # Create a command queue, i.e. the interface to the device
    mf = cl.mem_flags                   # Memory flags to specify the type of memory

    # Allocate memory on the device, buffer for the image
    image_buf = cl.Buffer(context, mf.WRITE_ONLY, image.nbytes)

    # Compile the kernel, build the program
    program = cl.Program(context, kernel_code).build()

    # Execute the kernel
    width, height = image.shape[1], image.shape[0]
    program.julia(queue, (width, height), None, 
                  np.float64(min_x), np.float64(max_x), np.float64(min_y), np.float64(max_y), 
                  image_buf, np.int32(width), np.int32(height), np.int32(iters)
                  )

    # Copy the result back to the host
    cl.enqueue_copy(queue, image, image_buf).wait()
    
    enable_output() # Restore the output __stdout__


def create_fractal_cupy(min_x, max_x, min_y, max_y, image, iters):
    """
    Create a Julia fractal on the GPU using CuPy's RawKernel.
    """
    # CUDA kernel as a string
    julia_kernel = cp.RawKernel(r"""
    extern "C" __global__
    void julia(
        double min_x, double max_x, double min_y, double max_y,
        unsigned char* image, int width, int height, int iters) {
        
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < width && y < height) {
            double pixel_size_x = (max_x - min_x) / width;
            double pixel_size_y = (max_y - min_y) / height;
            double real = min_x + x * pixel_size_x;
            double imag = min_y + y * pixel_size_y;
            double z_real = real;
            double z_imag = imag;
            double c_real = -0.7;
            double c_imag = 0.27015;

            int i;
            for (i = 0; i < iters; i++) {
                double z_real2 = z_real * z_real - z_imag * z_imag + c_real;
                double z_imag2 = 2.0 * z_real * z_imag + c_imag;
                z_real = z_real2;
                z_imag = z_imag2;
                if (z_real * z_real + z_imag * z_imag >= 4.0) {
                    break;
                }
            }
            image[y * width + x] = (unsigned char)(255.0 * i / iters);
        }
    }
    """, "julia")

    # Get the width and height of the image
    width, height = image.shape[1], image.shape[0]

    # Allocate memory on the GPU
    d_image = cp.zeros((height, width), dtype=cp.uint8)

    # Define the block and grid sizes
    threads_per_block = (16, 16)
    blocks_per_grid = (
        (width + threads_per_block[0] - 1) // threads_per_block[0],
        (height + threads_per_block[1] - 1) // threads_per_block[1],
    )

    # Launch the kernel
    julia_kernel(
        blocks_per_grid, threads_per_block,
        (cp.float64(min_x), cp.float64(max_x), cp.float64(min_y), cp.float64(max_y),
         d_image, cp.int32(width), cp.int32(height), cp.int32(iters))
    )

    # Copy the result back to the host
    np.copyto(image, d_image.get())  # Use d_image.get() to transfer the data to NumPy
