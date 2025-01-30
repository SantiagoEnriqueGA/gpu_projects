import cupy as cp
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

SIM_LENGTH = 60*3        # Simulation length in seconds
# SIM_LENGTH = 10          # Simulation length in seconds (for testing)
NUM_STEPS = int(SIM_LENGTH * FPS)  # Number of simulation steps

# Create timestamped output filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = f"particle_simulation_3d/vids/particle_simulation_{timestamp}.mp4"

# Initialize particle positions, velocities, and masses with pinned memory
positions = cp.asarray(cp.random.uniform(0, SPACE_SIZE, size=(NUM_PARTICLES, 3)))
positions = cp.array(positions, order='C')  # Ensure C-contiguous memory
velocities = cp.array(cp.random.uniform(-1, 1, size=(NUM_PARTICLES, 3)), order='C')
masses = cp.array(cp.random.uniform(1, 10, size=(NUM_PARTICLES, 1)), order='C')

# Pre-allocate arrays for visualization
positions_host = np.empty((NUM_PARTICLES, 3), dtype=np.float64)
velocities_host = np.empty((NUM_PARTICLES, 3), dtype=np.float64)
velocity_magnitudes_host = np.empty(NUM_PARTICLES, dtype=np.float64)

# Create CUDA streams for concurrent execution
stream1 = cp.cuda.Stream()
stream2 = cp.cuda.Stream()

def main():
    # Set up the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"Simulation: {NUM_PARTICLES:,} Colliding Particles", fontweight="bold")
    ax.set_xlim(0, SPACE_SIZE)
    ax.set_ylim(0, SPACE_SIZE)
    ax.set_zlim(0, SPACE_SIZE)
    
    # Remove axis labels and numbers
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.get_zaxis().set_visible(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])


    # Create scatter plot with initial empty data
    scat = ax.scatter([], [], [], s=2, c=[], cmap='winter')
    
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
        scat._offsets3d = (positions_host[:,0], positions_host[:,1], positions_host[:,2])
        scat.set_array(velocity_magnitudes_host)
        
        if step == 0:
            scat.set_clim(velocity_magnitudes_host.min(), velocity_magnitudes_host.max())
        
        return scat,

    # Create video writer
    writer = FFMpegWriter(fps=FPS, metadata={'title': 'Particle Simulation', 'artist': 'Matplotlib'}, bitrate=3600*3)

    # Save the animation
    with writer.saving(fig, OUTPUT_FILE, dpi=300):
        for step in tqdm(range(NUM_STEPS), desc="Simulating particles"):
            update(step)
            writer.grab_frame()

    print(f"Simulation saved as {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

# # Profiling the simulation
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
#     stats.dump_stats("particle_simulation_3d/prof/profile_results.prof")


