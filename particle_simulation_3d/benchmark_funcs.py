import time
import cupy as cp
import numpy as np

from sim import *
from cuda_compute_forces import compute_forces_cudaKernel 
from cuda_handle_particle_collisions import handle_particle_collisions_cudaKernel
from cuda_handle_boundary_collisions import handle_boundary_collisions_cudaKernel

# Constants
NUM_PARTICLES = 5000  # Number of particles
SPACE_SIZE = 200.0    # Size of the simulation space
DT = 0.005             # Time step
PARTICLE_RADIUS = 0.5  # Radius of particles for collision detection
ELASTICITY = 0.75     # Collision elasticity coefficient

N_RUNS = 10

# Initialize cupy particle positions, velocities, and masses 
positions_cp = cp.asarray(cp.random.uniform(0, SPACE_SIZE, size=(NUM_PARTICLES, 3)))
positions_cp = cp.array(positions, order='C')  # Ensure C-contiguous memory
velocities_cp = cp.array(cp.random.uniform(-1, 1, size=(NUM_PARTICLES, 3)), order='C')
masses_cp = cp.array(cp.random.uniform(1, 10, size=(NUM_PARTICLES, 1)), order='C')

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


# CuPy Specific Benchmarks
# -------------------------------------------------------------------------------------------------
from cupyx.profiler import benchmark

print(f"Benchmarking CuPy specific functions...")

n_warmup = 3
n_repeat = 100

print(benchmark(compute_forces_cudaKernel, (positions_cp, masses_cp), n_repeat=n_repeat, n_warmup=n_warmup))
print(benchmark(handle_particle_collisions_cudaKernel, (positions_cp, velocities_cp, masses_cp, PARTICLE_RADIUS, ELASTICITY), n_repeat=n_repeat, n_warmup=n_warmup))
print(benchmark(handle_boundary_collisions_cudaKernel, (positions_cp, velocities_cp, SPACE_SIZE, ELASTICITY), n_repeat=n_repeat, n_warmup=n_warmup))


# OUTPUT:
# Benchmarking 5,000 particles in a 200.00 x 200.00 space with 10 runs each.
# ----------------------------------------------------------------------------------------------------
# Benchmarking CuPy specific functions...
# compute_forces_cudaKernel:                CPU:   241.292 us   +/- 59.969 (min:   178.900 / max:   520.600) us     GPU-0:  9904.802 us   +/- 2514.454 (min:  8794.720 / max: 16142.879) us
# handle_particle_collisions_cudaKernel:    CPU:   205.574 us   +/- 48.420 (min:   123.200 / max:   396.700) us     GPU-0:  2788.102 us   +/- 48.173 (min:  2703.360 / max:  2969.184) us
# handle_boundary_collisions_cudaKernel:    CPU:    81.392 us   +/-  7.301 (min:    75.400 / max:   119.900) us     GPU-0:    92.203 us   +/-  8.553 (min:    84.544 / max:   131.072) us
