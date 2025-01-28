import time
import numpy as np
from numba import cuda
import matplotlib.pyplot as plt

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
    # Initialize arrays with initial conditions
    x = np.full(steps, x_init, dtype=np.float64)
    y = np.full(steps, y_init, dtype=np.float64)
    z = np.full(steps, z_init, dtype=np.float64)
    
    # Create device arrays
    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)
    d_z = cuda.to_device(z)
    
    # Create output arrays
    d_out_x = cuda.to_device(np.zeros(steps, dtype=np.float64))
    d_out_y = cuda.to_device(np.zeros(steps, dtype=np.float64))
    d_out_z = cuda.to_device(np.zeros(steps, dtype=np.float64))
    
    # Configure grid dimensions
    threads_per_block = 256//16
    blocks_per_grid = (steps + (threads_per_block - 1)) // threads_per_block
    
    # Run the simulation steps sequentially
    for step in range(steps - 1):  # -1 because we compute next step
        # Copy previous output to input (after first iteration)
        if step > 0:
            cuda.synchronize()
            d_x = cuda.to_device(out_x)
            d_y = cuda.to_device(out_y)
            d_z = cuda.to_device(out_z)
        
        # Run one step of the simulation
        lorenz_numba[blocks_per_grid, threads_per_block](
            d_x, d_y, d_z, s, r, b, dt, d_out_x, d_out_y, d_out_z
        )
        
        # Wait for GPU to finish and get results
        cuda.synchronize()
        out_x = d_out_x.copy_to_host()
        out_y = d_out_y.copy_to_host()
        out_z = d_out_z.copy_to_host()
    
    return out_x, out_y, out_z

def main():
    STEPS = 10_000              # Number of steps to run Lorenz attractor
    X = 0; Y = 1; Z = 1.001     # Initial values for x, y, z
    S = 10; R = 28; B = 2.667   # Constants for Lorenz attractor 
    DT = 0.01                   # Time step
    
    PLOT = True
    
    print("Running Lorenz attractor on CPU...           ", end="", flush=True)
    start_time = time.time()
    xs_cpu, ys_cpu, zs_cpu = run_lorenz_cpu(STEPS, X, Y, Z, S, R, B, DT)
    print(f" execution time: {time.time() - start_time:.4f} seconds")
    
    if check_numba_cuda():
        print("Running Lorenz attractor on GPU Numba CUDA...", end="", flush=True)
        start_time = time.time()
        xs_gpu, ys_gpu, zs_gpu = run_lorenz_gpu(STEPS, X, Y, Z, S, R, B, DT)
        print(f" execution time: {time.time() - start_time:.4f} seconds")

        
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
