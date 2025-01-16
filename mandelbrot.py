import numpy as np
from numba import jit, cuda
import matplotlib.pyplot as plt
from time import time

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        execution_time = end_time - start_time
        print(f"Execution time for {func.__name__} is {execution_time} seconds")
        return result
    return wrapper

def mandelbrot(x, y, max_iter):
    c = complex(x, y)
    z = 0.0j
    i = 0
    
    for i in range(max_iter):
        z = z*z + c
        if (z.real * z.real + z.imag * z.imag) >= 4:
            return i
    return 255

@jit
def mandelbrot_numba(x, y, max_iter):
    c = complex(x, y)
    z = 0.0j
    i = 0
    
    for i in range(max_iter):
        z = z*z + c
        if (z.real * z.real + z.imag * z.imag) >= 4:
            return i
    return 255
        
@timing_decorator        
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

@timing_decorator
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
        color = mandelbrot_numba(real, imag, iters)
        image[y, x] = color

size_multi = 2

image = np.zeros((500 * size_multi, 750 * size_multi), dtype=np.uint8)
create_fractal(-2.0, 1.0, -1.0, 1.0, image, 20)
plt.imshow(image)
plt.show()

image = np.zeros((500 * size_multi, 750 * size_multi), dtype=np.uint8)
threadsperblock = (16, 16)
blockspergrid_x = int(np.ceil(image.shape[1] / threadsperblock[0]))
blockspergrid_y = int(np.ceil(image.shape[0] / threadsperblock[1]))
blockspergrid = (blockspergrid_x, blockspergrid_y)
create_fractal_numba[blockspergrid, threadsperblock](-2.0, 1.0, -1.0, 1.0, image, 20)
plt.imshow(image)
plt.show()