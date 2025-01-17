import time
import numpy as np
from numba import cuda
import pyopencl as cl
import matplotlib.pyplot as plt
import os

from utils import timing_decorator, avg_timing_decorator
from utils import suppress_output, enable_output
from utils import check_numba_cuda, check_openCl


# -------------------------------------------------------------------------------------------------
# Functions to create the Mandelbrot fractal on the CPU and GPU
# -------------------------------------------------------------------------------------------------
def create_fractal_cpu(min_x, max_x, min_y, max_y, image, iters):
    """Create a Mandelbrot fractal on the CPU"""

    def mandelbrot(x, y, max_iter):
        """Calculate the Mandelbrot value for a given coordinate"""
        c = complex(x, y)           # Convert the x, y coordinates to a complex number
        z = 0.0j                    # Initialize z to 0
        
        # For each iteration, calculate z = z^2 + c
        for i in range(max_iter):   
            z = z * z + c
            if (z.real * z.real + z.imag * z.imag) >= 4:
                return i    # Return the number of iterations, i will be used to color the pixel
        return 255          # If the magnitude of z never exceeds 2, return 255 (white)
    
    # Set the width and height of the image
    width = image.shape[1]
    height = image.shape[0]

    # Set the pixel size in the x and y directions
    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    # For each pixel in the image, calculate the Mandelbrot value
    for x in range(width):
        # Real part of the complex number
        real = min_x + x * pixel_size_x
        for y in range(height):
            # Imaginary part of the complex number
            imag = min_y + y * pixel_size_y
            color = mandelbrot(real, imag, iters)
            image[y, x] = color

@cuda.jit
def create_fractal_numba(min_x, max_x, min_y, max_y, image, iters):
    """Create a Mandelbrot fractal on the GPU using Numba"""
    
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
        c = complex(real, imag) 
        z = 0.0j
        
        # For each iteration, calculate z = z^2 + c
        for i in range(iters):
            z = z * z + c
            if (z.real * z.real + z.imag * z.imag) >= 4:
                image[y, x] = i     # Set the pixel value to the number of iterations
                return
        image[y, x] = 255           # If the magnitude of z never exceeds, set the pixel value to 255 (white)

def create_fractal_opencl(min_x, max_x, min_y, max_y, image, iters):
    """Create a Mandelbrot fractal on the GPU using PyOpenCL"""
    # Set the environment variable
    os.environ['PYOPENCL_CTX'] = '0'
    
    # OpenCL kernel code
    # Similar to the CUDA kernel code, written in OpenCL C
    kernel_code = """
    __kernel void mandelbrot(
        const float min_x, const float max_x, const float min_y, const float max_y,
        __global uchar *image, const int width, const int height, const int iters) {
        
        int x = get_global_id(0);
        int y = get_global_id(1);
        
        if (x < width && y < height) {
            float pixel_size_x = (max_x - min_x) / width;
            float pixel_size_y = (max_y - min_y) / height;
            float real = min_x + x * pixel_size_x;
            float imag = min_y + y * pixel_size_y;
            float c_real = real;
            float c_imag = imag;
            float z_real = 0.0;
            float z_imag = 0.0;
            int i;
            for (i = 0; i < iters; i++) {
                float z_real2 = z_real * z_real - z_imag * z_imag + c_real;
                float z_imag2 = 2.0 * z_real * z_imag + c_imag;
                z_real = z_real2;
                z_imag = z_imag2;
                if (z_real * z_real + z_imag * z_imag >= 4.0) {
                    break;
                }
            }
            image[y * width + x] = i;
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
    program.mandelbrot(queue, (width, height), None, np.float32(min_x), np.float32(max_x), np.float32(min_y), np.float32(max_y), image_buf, np.int32(width), np.int32(height), np.int32(iters))

    # Copy the result back to the host
    cl.enqueue_copy(queue, image, image_buf).wait()
    
    enable_output() # Restore the output __stdout__



# -------------------------------------------------------------------------------------------------
# Functions to run and time the Mandelbrot fractal on the CPU and GPU
# -------------------------------------------------------------------------------------------------

# --------------------CPU--------------------
def run_cpu(size_multi=2, max_iters=50, show=True):
    width = 750 * size_multi
    height = 500 * size_multi
    
    image = np.zeros((height, width), dtype=np.uint8)
    
    start_time = time.time()
    create_fractal_cpu(-2.0, 1.0, -1.0, 1.0, image, max_iters)
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"Execution time for create_fractal is {execution_time:.3f} seconds")
    
    if show:
        plt.imshow(image)
        plt.title(f"Mandelbrot Set (CPU, iters={max_iters})")
        plt.show()
    return image

@avg_timing_decorator
def run_cpu_avg(size_multi=2, max_iters=50):
    width = 750 * size_multi
    height = 500 * size_multi
    image = np.zeros((height, width), dtype=np.uint8)
    create_fractal_cpu(-2.0, 1.0, -1.0, 1.0, image, max_iters)


# --------------------CUDA--------------------
def run_gpu_cuda(size_multi=2, max_iters=50, show=True):
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
    if show:
        plt.imshow(image)
        plt.title(f"Mandelbrot Set (GPU, iters={max_iters})")
        plt.show()
    return image

@avg_timing_decorator
def run_gpu_cuda_avg(size_multi=2, max_iters=50):
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



# --------------------OPENCL--------------------
def run_gpu_opencl(size_multi=2, max_iters=50, show=True):
    width = 750 * size_multi
    height = 500 * size_multi
    
    image = np.zeros((height, width), dtype=np.uint8)
    
    start_time = time.time()
    create_fractal_opencl(-2.0, 1.0, -1.0, 1.0, image, max_iters)
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"Execution time for create_fractal_opencl is {execution_time:.4f} seconds")
    
    if show:
        plt.imshow(image)
        plt.title(f"Mandelbrot Set (GPU with PyOpenCL, iters={max_iters})")
        plt.show()
    return image

@avg_timing_decorator
def run_gpu_opencl_avg(size_multi=2, max_iters=50):
    width = 750 * size_multi
    height = 500 * size_multi
    image = np.zeros((height, width), dtype=np.uint8)
    create_fractal_opencl(-2.0, 1.0, -1.0, 1.0, image, max_iters)



def main():
    # Parameters
    SIZE_MULTI = 2
    MAX_ITERS = 50
    SHOW_IMAGE = True
    CPU = True
    
    print(f"Creating Mandelbrot Set of size {750*SIZE_MULTI}x{500*SIZE_MULTI} with {MAX_ITERS} iterations")
    print(f"\tNumber of pixels:    {750*SIZE_MULTI*500*SIZE_MULTI:,}")
    print(f"\tPixels x Iterations: {750*SIZE_MULTI*500*SIZE_MULTI*MAX_ITERS:,}")
    print("-"*60)
    
    # Runs single and shows the Mandelbrot fractal 
    #--------------------------------------------------------------------------------------------
    if CPU:
        print("\nRunning CPU version...")
        run_cpu(size_multi=SIZE_MULTI, max_iters=MAX_ITERS, show=SHOW_IMAGE)
        
    # Check for CUDA-enabled GPU
    if check_numba_cuda():
        print("\nRunning GPU CUDA version...")    # Requires a CUDA-enabled GPU
        run_gpu_cuda(size_multi=SIZE_MULTI, max_iters=MAX_ITERS, show=SHOW_IMAGE)
    else:
        print("\n--CUDA-enabled GPU not available!")
    
    # Check for OpenCL-enabled GPU
    if check_openCl():
        print("\nRunning GPU OpenCL version...")  # Requires an OpenCL-enabled GPU
        run_gpu_opencl(size_multi=SIZE_MULTI, max_iters=MAX_ITERS, show=SHOW_IMAGE)
    else:
        print("\n--OpenCL-enabled GPU not available!")
    
    # Runs 5 times and calculates the average time taken to run the Mandelbrot fractal
    #--------------------------------------------------------------------------------------------
    if CPU:
        print("\nRunning CPU version (average timing)...")
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


if __name__ == "__main__":
    main()
