import time
import numpy as np
from numba import cuda
import matplotlib.pyplot as plt
from pytools import F

from utils import *
from lorenz_funcs import *

def run_lorenz_cpu(steps, x, y, z, s, r, b, dt):
    """Run the Lorenz attractor on the CPU."""
    # Initialize arrays to store the results
    xs = np.empty(steps).astype(np.float64)
    ys = np.empty(steps).astype(np.float64)
    zs = np.empty(steps).astype(np.float64)
    
    # Run the simulation, for each time step, update x, y, z
    for i in range(steps):
        x, y, z = lorenz_cpu(x, y, z, s, r, b, dt)
        xs[i], ys[i], zs[i] = x, y, z
    return xs, ys, zs

def run_lorenz_gpu(steps, x_init, y_init, z_init, s, r, b, dt):
    """Run the Lorenz attractor on the GPU."""
    # Initialize arrays to store the results, initial values for x, y, z
    x = np.full(steps, x_init).astype(np.float64)
    y = np.full(steps, y_init).astype(np.float64)
    z = np.full(steps, z_init).astype(np.float64)

    # Output arrays for x, y, z
    out_x, out_y, out_z = np.empty(steps), np.empty(steps), np.empty(steps)
    
    # Set up the GPU kernel
    threads_per_block = 256
    blocks_per_grid = (steps + (threads_per_block - 1)) // threads_per_block

    # Run the simulation on the GPU
    lorenz_numba[blocks_per_grid, threads_per_block](x, y, z, s, r, b, dt, out_x, out_y, out_z)
    return out_x, out_y, out_z

def run_lorenz_opencl(steps, x_init, y_init, z_init, s, r, b, dt):
    """Run the Lorenz attractor on the GPU using OpenCL."""
    return lorenz_opencl(steps, x_init, y_init, z_init, s, r, b, dt)

def main():
    STEPS = 10_000_000          # Number of steps to run Lorenz attractor
    X = 0; Y = 1; Z = 1.001     # Initial values for x, y, z
    S = 10; R = 28; B = 2.667   # Constants for Lorenz attractor 
    DT = 0.01                   # Time step
    
    PLOT = False
    
    print("Running Lorenz attractor on CPU...")
    start_time = time.time()
    xs_cpu, ys_cpu, zs_cpu = run_lorenz_cpu(STEPS, X, Y, Z, S, R, B, DT)
    print(f"CPU execution time: {time.time() - start_time:.4f} seconds")
    
    if check_numba_cuda():
        print("Running Lorenz attractor on GPU Numba CUDA...")
        start_time = time.time()
        xs_gpu, ys_gpu, zs_gpu = run_lorenz_gpu(STEPS, X, Y, Z, S, R, B, DT)
        print(f"GPU execution time: {time.time() - start_time:.4f} seconds")

    if check_openCl():
        print("Running Lorenz attractor on GPU OpenCL...")
        start_time = time.time()
        xs_gpu, ys_gpu, zs_gpu = run_lorenz_opencl(STEPS, X, Y, Z, S, R, B, DT)
        print(f"GPU execution time: {time.time() - start_time:.4f} seconds")
        
    if PLOT:
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(121, projection='3d')
        ax.plot(xs_cpu, ys_cpu, zs_cpu, lw=0.5)
        ax.set_title("Lorenz Attractor (CPU)")
        ax = fig.add_subplot(122, projection='3d')
        ax.plot(xs_gpu, ys_gpu, zs_gpu, lw=0.5)
        ax.set_title("Lorenz Attractor (GPU)")
        plt.show()


if __name__ == "__main__":
    main()
