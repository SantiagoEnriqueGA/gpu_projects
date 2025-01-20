import time
import cupy as cp
import numpy as np

from sim import *
from cuda_compute_forces import compute_forces_cudaKernel 
from cuda_handle_particle_collisions import handle_particle_collisions_cudaKernel

# Constants
NUM_PARTICLES = 5000  # Number of particles
SPACE_SIZE = 200.0    # Size of the simulation space
DT = 0.005             # Time step
PARTICLE_RADIUS = 0.5  # Radius of particles for collision detection
ELASTICITY = 0.75     # Collision elasticity coefficient

N_RUNS = 10

# Initialize cupy particle positions, velocities, and masses 
positions_cp = cp.random.uniform(0, SPACE_SIZE, size=(NUM_PARTICLES, 2))
velocities_cp = cp.random.uniform(-1, 1, size=(NUM_PARTICLES, 2))
masses_cp = cp.random.uniform(1, 10, size=(NUM_PARTICLES, 1))

# Initialize numpy particle positions, velocities, and masses
positions_np = positions_cp.get()
velocities_np = velocities_cp.get()
masses_np = masses_cp.get()

# Affirm that cupy and numpy arrays are the same
assert isinstance(positions_np, np.ndarray)
assert isinstance(velocities_np, np.ndarray)
assert isinstance(masses_np, np.ndarray)    
assert np.allclose(positions_np, positions_cp.get())
assert np.allclose(velocities_np, velocities_cp.get())
assert np.allclose(masses_np, masses_cp.get())


def benchmark_function(func, *args, repeats=10):
    """Benchmark a function by timing its execution over multiple runs."""
    times = []
    for _ in range(repeats):
        start_time = time.time()
        func(*args)
        cp.cuda.Stream.null.synchronize()  # Ensure GPU computations are done
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = np.mean(times)
    std_dev = np.std(times)
    return avg_time, std_dev


print(f"Benchmarking {NUM_PARTICLES:,} particles in a {SPACE_SIZE:.2f} x {SPACE_SIZE:.2f} space with {N_RUNS} runs each.")
print("-"*100)

# Benchmark compute_forces
# -------------------------------------------------------------------------------------------------
print(f"\nBenchmarking gravitational force computations...")

np_time, np_std = benchmark_function(compute_forces_np, positions_cp, masses_cp, repeats=N_RUNS)
print(f"compute_forces_np average time:                         {np_time:.5f} seconds", f"(std_dev: {np_std:.5f} seconds)")

cp_time, cp_std = benchmark_function(compute_forces_cp, positions_cp, masses_cp, repeats=N_RUNS)
print(f"compute_forces_cp average time:                         {cp_time:.5f} seconds", f"(std_dev: {cp_std:.5f} seconds)")

cuda_time, cuda_std = benchmark_function(compute_forces_cudaKernel, positions_cp, masses_cp, repeats=N_RUNS)
print(f"compute_forces_cudaKernel average time:                 {cuda_time:.5f} seconds", f"(std_dev: {cuda_std:.5f} seconds)")

min_time = min(np_time, cp_time, cuda_time)
max_time = max(np_time, cp_time, cuda_time)
speedup = max_time / min_time
print("-"*100)
print(f"Speedup:                                                {speedup:.2f}x")


# Benchmark handle_particle_collisions
# -------------------------------------------------------------------------------------------------
print(f"\nBenchmarking particle-particle collision handling...")

np_time, np_std = benchmark_function(handle_particle_collisions_np, positions_cp, velocities_cp, masses_cp, repeats=N_RUNS)
print(f"handle_particle_collisions_np average time:             {np_time:.5f} seconds", f"(std_dev: {np_std:.5f} seconds)")

cp_time, cp_std = benchmark_function(handle_particle_collisions_cp, positions_cp, velocities_cp, masses_cp, repeats=N_RUNS)
print(f"handle_particle_collisions_cp average time:             {cp_time:.5f} seconds", f"(std_dev: {cp_std:.5f} seconds)")

cKDTree_time, cKDTree_std = benchmark_function(handle_particle_collisions_cKDTree, positions_cp, velocities_cp, masses_cp, repeats=N_RUNS)
print(f"handle_particle_collisions_cKDTree average time:        {cKDTree_time:.5f} seconds", f"(std_dev: {cKDTree_std:.5f} seconds)")

cuda_time, cuda_std = benchmark_function(handle_particle_collisions_cudaKernel, positions_cp, velocities_cp, masses_cp, PARTICLE_RADIUS, ELASTICITY, repeats=N_RUNS)
print(f"handle_particle_collisions_cudaKernel average time:     {cuda_time:.5f} seconds", f"(std_dev: {cuda_std:.5f} seconds)")

min_time = min(np_time, cp_time, cKDTree_time, cuda_time)
max_time = max(np_time, cp_time, cKDTree_time, cuda_time)
speedup = max_time / min_time
print("-"*100)
print(f"Speedup:                                                {speedup:.2f}x")


# Benchmark handle_boundary_collisions
# -------------------------------------------------------------------------------------------------
print("")
boundary_time, boundary_std = benchmark_function(handle_boundary_collisions, positions_cp, velocities_cp, repeats=N_RUNS)
print(f"handle_boundary_collisions average time:                {boundary_time:.5f} seconds", f"(std_dev: {boundary_std:.5f} seconds)")
