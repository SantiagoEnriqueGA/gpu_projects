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

# Initialize particle positions, velocities, and masses with pinned memory
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
print("-"*100)
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
print("-"*100)
print("Passed! all collisions and velocities are equal.")

# Compare boundary collision handling
# -------------------------------------------------------------------------------------------------
print("\nComparing boundary collision handling...")

boundary_collisions = handle_boundary_collisions(positions_np, velocities_np)
cuda_boundary_collisions = handle_boundary_collisions_cudaKernel(positions_cp, velocities_cp, SPACE_SIZE, ELASTICITY)

print(f"   First elements of boundary_collisions (positions):       {boundary_collisions[0][:1]}")
print(f"   First elements of cuda_boundary_collisions (positions):  {cuda_boundary_collisions[0][:1]}")
print("")
print(f"   First elements of boundary_collisions (velocities):      {boundary_collisions[1][:1]}")
print(f"   First elements of cuda_boundary_collisions (velocities): {cuda_boundary_collisions[1][:1]}")


assert np.allclose(boundary_collisions[0], cuda_boundary_collisions[0], rtol=1e-5, atol=1e-5)
print("-"*100)
print("Passed! all collisions and velocities are equal.")

# OUTPUT:
# Comparing gravitational force computations...
#    First elements of np_total_forces:   [[-22.71349774  23.03124566]]
#    First elements of cp_total_forces:   [[-22.71349774  23.03124566]]
#    First elements of cuda_total_forces: [[-22.71349774  23.03124566]]
# ----------------------------------------------------------------------------------------------------
# Passed! all forces are equal.

# Comparing particle-particle collision handling...
#    First elements of np_collisions (positions):      [[136.1391209  177.98479804]]
#    First elements of cp_collisions (positions):      [[136.1391209  177.98479804]]
#    First elements of cKDTree_collisions (positions): [[136.1391209  177.98479804]]
#    First elements of cuda_collisions (positions):    [[136.1391209  177.98479804]]

#    First elements of np_collisions (velocities):      [[0.06512051 0.71643243]]
#    First elements of cp_collisions (velocities):      [[0.06512051 0.71643243]]
#    First elements of cKDTree_collisions (velocities): [[0.06512051 0.71643243]]
#    First elements of cuda_collisions (velocities):    [[0.06512051 0.71643243]]
# ----------------------------------------------------------------------------------------------------
# Passed! all collisions and velocities are equal.

# Comparing boundary collision handling...
#    First elements of boundary_collisions (positions):       [[136.1391209  177.98479804]]
#    First elements of cuda_boundary_collisions (positions):  [[136.1391209  177.98479804]]

#    First elements of boundary_collisions (velocities):      [[0.06512051 0.71643243]]
#    First elements of cuda_boundary_collisions (velocities): [[0.06512051 0.71643243]]
# ----------------------------------------------------------------------------------------------------
# Passed! all collisions and velocities are equal.