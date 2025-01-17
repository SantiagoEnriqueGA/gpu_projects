import time
import numpy as np
from numba import cuda
import matplotlib.pyplot as plt

def mandelbrot(x, y, max_iter):
    c = complex(x, y)
    z = 0.0j
    for i in range(max_iter):
        z = z * z + c
        if (z.real * z.real + z.imag * z.imag) >= 4:
            return i
    return 255

def create_fractal(min_x, max_x, min_y, max_y, image, iters):
    width = image.shape[1]
    height = image.shape[0]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    for x in range(width):
        real = min_x + x * pixel_size_x
        for y in range(height):
            imag = min_y + y * pixel_size_y
            color = mandelbrot(real, imag, iters)
            image[y, x] = color


@cuda.jit
def create_fractal_numba(min_x, max_x, min_y, max_y, image, iters):
    width = image.shape[1]
    height = image.shape[0]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    x, y = cuda.grid(2)

    if x < width and y < height:
        real = min_x + x * pixel_size_x
        imag = min_y + y * pixel_size_y
        c = complex(real, imag)
        z = 0.0j
        for i in range(iters):
            z = z * z + c
            if (z.real * z.real + z.imag * z.imag) >= 4:
                image[y, x] = i
                return
        image[y, x] = 255


def main():
    # Adjustable parameters
    size_multi = 100             # Scale the resolution
    width = 750 * size_multi    # Image width
    height = 500 * size_multi   # Image height
    max_iters = 50              # Number of iterations for Mandelbrot calculation
    
    print(f"Creating Mandelbrot Set of size {width}x{height} with {max_iters} iterations")
    print("-"*60)

    # CPU computation
    image = np.zeros((height, width), dtype=np.uint8)
    start_time = time.time()
    create_fractal(-2.0, 1.0, -1.0, 1.0, image, max_iters)
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution time for create_fractal is {execution_time:.6f} seconds")
    
    plt.imshow(image)
    plt.title(f"Mandelbrot Set (CPU, iters={max_iters})")
    plt.show()

    # GPU computation
    image = np.zeros((height, width), dtype=np.uint8)
    d_image = cuda.to_device(image)  # Copy to device

    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(width / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(height / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    start_time = time.time()
    create_fractal_numba[blockspergrid, threadsperblock](-2.0, 1.0, -1.0, 1.0, d_image, max_iters)
    cuda.synchronize()  # Wait for kernel to complete
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution time for create_fractal_numba is {execution_time:.6f} seconds")

    image = d_image.copy_to_host()  # Copy back to host
    plt.imshow(image)
    plt.title(f"Mandelbrot Set (GPU, iters={max_iters})")
    plt.show()

if __name__ == "__main__":
    main()
