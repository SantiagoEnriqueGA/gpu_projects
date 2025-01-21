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


# Compare gravitational force computations
# -------------------------------------------------------------------------------------------------
print("Comparing gravitational force computations...")

np_total_forces = compute_forces_np(positions_np, masses_np)
cp_total_forces = compute_forces_cp(positions_cp, masses_cp)
cuda_total_forces = compute_forces_cudaKernel(positions_cp, masses_cp)

print(f"   First elements of np_total_forces:   {np_total_forces[:1]}")
print(f"   First elements of cp_total_forces:   {cp_total_forces[:1]}")
print(f"   First elements of cuda_total_forces: {cuda_total_forces[:1]}")

assert np.allclose(np_total_forces, cp_total_forces, rtol=1e-5, atol=1e-5)
assert np.allclose(np_total_forces, cuda_total_forces, rtol=1e-5, atol=1e-5)
print("-"*80)
print("Passed! all forces are equal.")

# Compare particle-particle collision handling
# -------------------------------------------------------------------------------------------------
print("\nComparing particle-particle collision handling...")

np_collisions = handle_particle_collisions_np(positions_np, velocities_np, masses_np)
cp_collisions = handle_particle_collisions_cp(positions_cp, velocities_cp, masses_cp)
cKDTree_collisions = handle_particle_collisions_cKDTree(positions_cp, velocities_cp, masses_cp)
cuda_collisions = handle_particle_collisions_cudaKernel(positions_cp, velocities_cp, masses_cp, PARTICLE_RADIUS, ELASTICITY)

print(f"   First elements of np_collisions (positions):      {np_collisions[0][:1]}")
print(f"   First elements of cp_collisions (positions):      {cp_collisions[0][:1]}")
print(f"   First elements of cKDTree_collisions (positions): {cKDTree_collisions[0][:1]}")
print(f"   First elements of cuda_collisions (positions):    {cuda_collisions[0][:1]}")
print("")
print(f"   First elements of np_collisions (velocities):      {np_collisions[1][:1]}")
print(f"   First elements of cp_collisions (velocities):      {cp_collisions[1][:1]}")
print(f"   First elements of cKDTree_collisions (velocities): {cKDTree_collisions[1][:1]}")
print(f"   First elements of cuda_collisions (velocities):    {cuda_collisions[1][:1]}")

assert np.allclose(np_collisions[0], cp_collisions[0], rtol=1e-5, atol=1e-5)
assert np.allclose(np_collisions[0], cKDTree_collisions[0], rtol=1e-5, atol=1e-5)
assert np.allclose(np_collisions[0], cuda_collisions[0], rtol=1e-5, atol=1e-5)
print("-"*80)
print("Passed! all collisions and velocities are equal.")


# OUTPUT:
# Comparing gravitational force computations...
#    First elements of np_total_forces:   [[ -0.08247896 -16.43854861]]
#    First elements of cp_total_forces:   [[ -0.08247896 -16.43854861]]
#    First elements of cuda_total_forces: [[ -0.08247896 -16.43854861]]
# --------------------------------------------------------------------------------
# Passed! all forces are equal.

# Comparing particle-particle collision handling...
#    First elements of np_collisions (positions):      [[97.55379889 88.26877469]]
#    First elements of cp_collisions (positions):      [[97.55379889 88.26877469]]
#    First elements of cKDTree_collisions (positions): [[97.55379889 88.26877469]]
#    First elements of cuda_collisions (positions):    [[97.55379889 88.26877469]]

#    First elements of np_collisions (velocities):      [[0.23632976 0.80954981]] 
#    First elements of cp_collisions (velocities):      [[0.23632976 0.80954981]] 
#    First elements of cKDTree_collisions (velocities): [[0.23632976 0.80954981]] 
#    First elements of cuda_collisions (velocities):    [[0.23632976 0.80954981]] 
# --------------------------------------------------------------------------------
# Passed! all collisions and velocities are equal.
