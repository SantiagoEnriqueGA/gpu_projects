import time
import numpy as np
from numba import cuda
import matplotlib.pyplot as plt

from utils import *
from lorenz_funcs import *
from lorenz import run_lorenz_cpu, run_lorenz_gpu

from numba import NumbaPerformanceWarning
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

def main():
    X = 0; Y = 1; Z = 1.001     # Initial values for x, y, z
    S = 10; R = 28; B = 2.667   # Constants for Lorenz attractor 
    DT = 0.01                   # Time step
    

    STEPS = [1_000, 5_000, 10_000, 15_000, 20_000, 25_000, 30_000, 35_000, 40_000]
    cpu_times = []
    gpu_times = []

    for steps in STEPS:
        print(f"Running {steps} steps of Lorenz attractor... ")

        print("\tRunning CPU version...        ", end="", flush=True)
        start_time = time.time()
        xs_cpu, ys_cpu, zs_cpu = run_lorenz_cpu(steps, X, Y, Z, S, R, B, DT)
        t = time.time() - start_time
        cpu_times.append(t)
        print(f" execution time: {t:.4f} seconds")
        
        if check_numba_cuda():
            print("\tRunning GPU CUDA version...  ", end="", flush=True)
            start_time = time.time()
            xs_gpu, ys_gpu, zs_gpu = run_lorenz_gpu(steps, X, Y, Z, S, R, B, DT)
            t = time.time() - start_time
            gpu_times.append(t)
            print(f" execution time: {t:.4f} seconds")
        
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(STEPS, cpu_times, label="CPU")
    plt.plot(STEPS, gpu_times, label="GPU")
    plt.xlabel("Number of steps")
    plt.ylabel("Execution time (seconds)")
    plt.title("Execution time of Lorenz Attractor")
    plt.legend()
    plt.tight_layout()
    plt.show()
        

if __name__ == "__main__":
    main()
