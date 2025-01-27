import time
import numpy as np
from numba import cuda

from utils import *
from julia_funcs import *


# Parameters
SIZE_MULTI = 1
MAX_ITERS = 50

# Run all Julia Set functions
# -------------------------------------------------------------------------------------------------
print(f"Creating Julia Set Set of size {750*SIZE_MULTI}x{500*SIZE_MULTI} with {MAX_ITERS} iterations")
print(f"\tNumber of pixels:    {750*SIZE_MULTI*500*SIZE_MULTI:,}")
print(f"\tPixels x Iterations: {750*SIZE_MULTI*500*SIZE_MULTI*MAX_ITERS:,}")
print("-"*60)

width = 750 * SIZE_MULTI
height = 500 * SIZE_MULTI

print("Generating Julia Set fractal (CPU)...")
image = np.zeros((height, width), dtype=np.uint8)
create_fractal_cpu(-2.0, 1.0, -1.0, 1.0, image, MAX_ITERS)
cpu_image = image.copy()

if check_numba_cuda():
    print("\nGenerating Julia Set fractal (GPU Numba CUDA)...")
    image = np.zeros((height, width), dtype=np.uint8)
    d_image = cuda.to_device(image)
    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(width / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(height / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    create_fractal_numba[blockspergrid, threadsperblock](-2.0, 1.0, -1.0, 1.0, d_image, MAX_ITERS)
    cuda.synchronize()
    numba_image = d_image.copy_to_host()


if check_openCl():
    print("\nGenerating Julia Set fractal (GPU OpenCL)...")
    image = np.zeros((height, width), dtype=np.uint8)
    create_fractal_opencl(-2.0, 1.0, -1.0, 1.0, image, MAX_ITERS)
    opencl_image = image.copy()


if check_cupy():
    print("\nGenerating Julia Set fractal (GPU CuPy)...")
    image = np.zeros((height, width), dtype=np.uint8)
    create_fractal_cupy(-2.0, 1.0, -1.0, 1.0, image, MAX_ITERS)
    cupy_image = image.copy()


# Compare the results
# -------------------------------------------------------------------------------------------------
print("\nComparing results...")

print(f"CPU:    {np.unique(cpu_image).shape[0]} unique pixels")
print(f"Numba:  {np.unique(numba_image).shape[0]} unique pixels")
print(f"OpenCL: {np.unique(opencl_image).shape[0]} unique pixels")
print(f"CuPy:   {np.unique(cupy_image).shape[0]} unique pixels")

print("\nMax and Min Values:")
print(f"CPU:    (max: {np.max(cpu_image)}, min: {np.min(cpu_image)})")
print(f"Numba:  (max: {np.max(numba_image)}, min: {np.min(numba_image)})")
print(f"OpenCL: (max: {np.max(opencl_image)}, min: {np.min(opencl_image)})")
print(f"CuPy:   (max: {np.max(cupy_image)}, min: {np.min(cupy_image)})")    



import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

plt.hist(cpu_image.ravel(), bins=256, range=(0, 255), alpha=0.5, label="CPU")
plt.hist(numba_image.ravel(), bins=256, range=(0, 255), alpha=0.5, label="Numba")
plt.hist(opencl_image.ravel(), bins=256, range=(0, 255), alpha=0.5, label="OpenCL")
plt.hist(cupy_image.ravel(), bins=256, range=(0, 255), alpha=0.5, label="CuPy")

plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.title("Histogram Comparison of Julia Set Fractals")
plt.legend()
plt.show()


plt.figure(figsize=(16, 8))

plt.subplot(2, 2, 1)
plt.imshow(cpu_image, cmap='hot', extent=(-2, 1, -1, 1))
plt.title("CPU")

plt.subplot(2, 2, 2)
plt.imshow(numba_image, cmap='hot', extent=(-2, 1, -1, 1))
plt.title("Numba")

plt.subplot(2, 2, 3)
plt.imshow(opencl_image, cmap='hot', extent=(-2, 1, -1, 1))
plt.title("OpenCL")

plt.subplot(2, 2, 4)
plt.imshow(cupy_image, cmap='hot', extent=(-2, 1, -1, 1))
plt.title("CuPy")

plt.tight_layout()
plt.show()
