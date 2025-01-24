import cupy as cp
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm
from datetime import datetime

# CUDA kernels
from cuda_compute_forces import compute_forces_cudaKernel 
from cuda_handle_particle_collisions import handle_particle_collisions_cudaKernel
from cuda_handle_boundary_collisions import handle_boundary_collisions_cudaKernel

# Constants
FPS = 60                # Frames per second
NUM_PARTICLES = 2000  # Number of particles
SPACE_SIZE = 200.0    # Size of the simulation space
DT = 0.025            # Time step
G = 0.0               # Gravitational constant
ELASTICITY = 0.75     # Collision elasticity coefficient
PARTICLE_RADIUS = 0.5  # Radius of particles for collision detection

# SIM_LENGTH = 60*3        # Simulation length in seconds
SIM_LENGTH = 5        # Simulation length in seconds (for testing)
NUM_STEPS = int(SIM_LENGTH * FPS)  # Number of simulation steps

# Create timestamped output filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = f"particle_simulation/vids/particle_simulation_{timestamp}.mp4"

# Initialize particle positions, velocities, and masses with pinned memory
positions = cp.asarray(cp.random.uniform(0, SPACE_SIZE, size=(NUM_PARTICLES, 2)))
positions = cp.array(positions, order='C')  # Ensure C-contiguous memory
velocities = cp.array(cp.random.uniform(-1, 1, size=(NUM_PARTICLES, 2)), order='C')
masses = cp.array(cp.random.uniform(1, 10, size=(NUM_PARTICLES, 1)), order='C')

# Pre-allocate arrays for visualization
positions_host = np.empty((NUM_PARTICLES, 2), dtype=np.float64)
velocities_host = np.empty((NUM_PARTICLES, 2), dtype=np.float64)
velocity_magnitudes_host = np.empty(NUM_PARTICLES, dtype=np.float64)

# Create CUDA streams for concurrent execution
stream1 = cp.cuda.Stream()
stream2 = cp.cuda.Stream()

def main():
    # Set up the plot
    fig, ax = plt.subplots()
    ax.set_title(f"Particle Simulation: {NUM_PARTICLES:,} particles", fontweight="bold")
    ax.set_xlim(0, SPACE_SIZE)
    ax.set_ylim(0, SPACE_SIZE)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Create scatter plot with initial empty data
    scat = ax.scatter([], [], s=2, c=[], cmap='winter')
    
    # Create a colorbar 
    cbar = plt.colorbar(scat, label='Particle Velocity', aspect=40)
    cbar.ax.set_ylabel('Particle Velocity', fontweight='bold')
    cbar.ax.tick_params(labelsize=0)    # Remove tick labels
    cbar.set_ticks([])                  # Remove tick marks


    # Update function for the animation
    def update(step):
        global positions, velocities
        
        # Use streams for concurrent execution of physics calculations
        with stream1:
            forces = compute_forces_cudaKernel(positions, masses, G=G)
            velocities += (forces / masses) * DT
        
        with stream2:
            positions += velocities * DT
            positions, velocities = handle_particle_collisions_cudaKernel(
                positions, velocities, masses, 
                particle_radius=PARTICLE_RADIUS, 
                elasticity=ELASTICITY
            )
            positions, velocities = handle_boundary_collisions_cudaKernel(
                positions, velocities, space_size=SPACE_SIZE, elasticity=ELASTICITY
            )
        
        # Synchronize streams before visualization
        stream1.synchronize()
        stream2.synchronize()
        
        # Transfer data to pre-allocated CPU arrays
        cp.asnumpy(positions, out=positions_host)
        cp.asnumpy(velocities, out=velocities_host)
        
        # Calculate velocity magnitudes on CPU
        velocity_magnitudes_host[:] = np.linalg.norm(velocities_host, axis=1)
        
        # Update visualization
        scat.set_offsets(positions_host)
        scat.set_array(velocity_magnitudes_host)
        
        if step == 0:
            scat.set_clim(velocity_magnitudes_host.min(), velocity_magnitudes_host.max())
        
        return scat,

    # Create video writer
    writer = FFMpegWriter(fps=FPS, metadata={'title': 'Particle Simulation', 'artist': 'Matplotlib'}, bitrate=3600)

    # Save the animation
    with writer.saving(fig, OUTPUT_FILE, dpi=300):
        for step in tqdm(range(NUM_STEPS), desc="Simulating particles"):
            update(step)
            writer.grab_frame()

    print(f"Simulation saved as {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

# Profiling the simulation
# if __name__ == "__main__":
#     import cProfile
#     import pstats

#     profiler = cProfile.Profile()
#     profiler.enable()
    
#     main()
    
#     profiler.disable()
#     stats = pstats.Stats(profiler)
#     stats.strip_dirs()
#     stats.sort_stats('cumtime')  # Sort by cumulative time
#     stats.print_stats(20)  # Print the top 20 results
#     stats.dump_stats("particle_simulation/prof/profile_results.prof")



# -------------------------------------------------------------------------------------------------
# Functions no longer used - see benchmark_funcs.py
# -------------------------------------------------------------------------------------------------

# Functions to handle particle-particle collisions
# -------------------------------------------------------------------------------------------------
def handle_particle_collisions_cp(positions, velocities, masses):
    """CuPy implementation of particle-particle collision handling"""
    delta = positions[:, cp.newaxis, :] - positions[cp.newaxis, :, :]
    distances = cp.linalg.norm(delta, axis=2)
    
    # Find colliding pairs (excluding self-collisions)
    colliding_pairs = cp.argwhere(
        (distances < 2 * PARTICLE_RADIUS) & 
        (distances > 0)  # Exclude self-collisions
    )
    
    for i, j in colliding_pairs:
        # Calculate collision normal and tangent
        normal = (positions[i] - positions[j]) / distances[i, j]
        
        # Relative velocity
        rel_velocity = velocities[i] - velocities[j]
        
        # Normal component of relative velocity
        vel_normal = cp.dot(rel_velocity, normal)
        
        # Only process collision if particles are moving toward each other
        if vel_normal < 0:
            # Masses
            m1, m2 = masses[i, 0], masses[j, 0]
            
            # Calculate new velocities using conservation of momentum and energy
            new_vel_i = velocities[i] - (2 * m2 / (m1 + m2)) * vel_normal * normal
            new_vel_j = velocities[j] + (2 * m1 / (m1 + m2)) * vel_normal * normal
            
            # Apply elasticity
            velocities[i] = ELASTICITY * new_vel_i + (1 - ELASTICITY) * velocities[i]
            velocities[j] = ELASTICITY * new_vel_j + (1 - ELASTICITY) * velocities[j]
            
            # Separate overlapping particles
            overlap = 2 * PARTICLE_RADIUS - distances[i, j]
            if overlap > 0:
                separation = overlap * normal / 2
                positions[i] += separation
                positions[j] -= separation
    
    return positions, velocities


def handle_particle_collisions_np(positions, velocities, masses):
    """NumPy implementation of particle-particle collision handling"""
    delta = positions[:, np.newaxis, :] - positions[cp.newaxis, :, :]
    distances = np.linalg.norm(delta, axis=2)
    
    # Find colliding pairs (excluding self-collisions)
    colliding_pairs = np.argwhere(
        (distances < 2 * PARTICLE_RADIUS) & 
        (distances > 0)  # Exclude self-collisions
    )
    
    for i, j in colliding_pairs:
        # Calculate collision normal and tangent
        normal = (positions[i] - positions[j]) / distances[i, j]
        
        # Relative velocity
        rel_velocity = velocities[i] - velocities[j]
        
        # Normal component of relative velocity
        vel_normal = np.dot(rel_velocity, normal)
        
        # Only process collision if particles are moving toward each other
        if vel_normal < 0:
            # Masses
            m1, m2 = masses[i, 0], masses[j, 0]
            
            # Calculate new velocities using conservation of momentum and energy
            new_vel_i = velocities[i] - (2 * m2 / (m1 + m2)) * vel_normal * normal
            new_vel_j = velocities[j] + (2 * m1 / (m1 + m2)) * vel_normal * normal
            
            # Apply elasticity
            velocities[i] = ELASTICITY * new_vel_i + (1 - ELASTICITY) * velocities[i]
            velocities[j] = ELASTICITY * new_vel_j + (1 - ELASTICITY) * velocities[j]
            
            # Separate overlapping particles
            overlap = 2 * PARTICLE_RADIUS - distances[i, j]
            if overlap > 0:
                separation = overlap * normal / 2
                positions[i] += separation
                positions[j] -= separation
    
    return positions, velocities

def handle_particle_collisions_cKDTree(positions, velocities, masses):
    """cKDTree implementation of particle-particle collision handling"""
    tree = cKDTree(cp.asnumpy(positions))
    pairs = tree.query_pairs(r=2 * PARTICLE_RADIUS)

    for i, j in pairs:
        i, j = cp.int32(i), cp.int32(j)  # Convert to GPU-compatible indices
        delta = positions[i] - positions[j]
        dist = cp.linalg.norm(delta)
        if dist > 0:
            normal = delta / dist
            rel_velocity = velocities[i] - velocities[j]
            vel_normal = cp.dot(rel_velocity, normal)
            if vel_normal < 0:
                m1, m2 = masses[i, 0], masses[j, 0]
                new_vel_i = velocities[i] - (2 * m2 / (m1 + m2)) * vel_normal * normal
                new_vel_j = velocities[j] + (2 * m1 / (m1 + m2)) * vel_normal * normal
                velocities[i], velocities[j] = new_vel_i, new_vel_j
    
    return positions, velocities

# Function to compute pairwise gravitational forces
# -------------------------------------------------------------------------------------------------
def compute_forces_np(positions, masses, G=1.0, epsilon=1e-5):
    """NumPy implementation of gravitational force computation"""
    delta = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    distances = np.sqrt(np.sum(delta * delta, axis=2) + epsilon)
    forces_magnitude = G * (masses * masses.T) / (distances * distances)
    forces = forces_magnitude[:, :, np.newaxis] * (delta / distances[:, :, np.newaxis])
    # Zero out self-interactions
    forces[np.arange(len(positions)), np.arange(len(positions))] = 0
    total_forces = np.sum(forces, axis=1)
    return total_forces

def compute_forces_cp(positions, masses, G=1.0, epsilon=1e-5):
    """CuPy implementation of gravitational force computation"""
    delta = positions[:, cp.newaxis, :] - positions[cp.newaxis, :, :]
    distances = cp.sqrt(cp.sum(delta * delta, axis=2) + epsilon)
    forces_magnitude = G * (masses * masses.T) / (distances * distances)
    forces = forces_magnitude[:, :, cp.newaxis] * (delta / distances[:, :, cp.newaxis])
    # Zero out self-interactions
    forces[cp.arange(len(positions)), cp.arange(len(positions))] = 0
    total_forces = cp.sum(forces, axis=1)
    return total_forces


# Function to handle boundary collisions
# -------------------------------------------------------------------------------------------------
def handle_boundary_collisions(positions, velocities):
    """Handle boundary collisions, bouncing particles off the walls."""
    for i in range(2):
        out_of_bounds_low = positions[:, i] < 0
        out_of_bounds_high = positions[:, i] > SPACE_SIZE
        velocities[out_of_bounds_low | out_of_bounds_high, i] *= -ELASTICITY
        positions[out_of_bounds_low, i] = 0
        positions[out_of_bounds_high, i] = SPACE_SIZE
    return positions, velocities