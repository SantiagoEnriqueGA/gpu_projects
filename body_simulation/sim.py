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

# Constants
FPS = 60               # Frames per second
SPACE_SIZE = 200       # Size of the simulation space
DT = 0.025             # Time step
G = 0.5                # Gravitational constant
ELASTICITY = 0.75      # Collision elasticity coefficient

# SIM_LENGTH = 60*3        # Simulation length in seconds
SIM_LENGTH = 5        # Simulation length in seconds (for testing)
NUM_STEPS = int(SIM_LENGTH * FPS)  # Number of simulation steps

# Create timestamped output filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = f"body_simulation/vids/particle_simulation_{timestamp}.mp4"

# Initialize particle positions, velocities, masses, and sizes with pinned memory
# 3 Different particles of different masses, sizes arround a central point (most massive)
sizes = cp.array([5, 1, 2, 3], dtype=cp.float64)
masses = cp.array([100, 1, 10, 50], dtype=cp.float64)
positions = cp.array([
    [100, 100],  # Central particle
    [70, 100],   # Smallest particle
    [120, 100],  # Medium particle
    [100, 120]   # Largest particle
], dtype=cp.float64)
velocities = cp.array([
    [0, 0],  # Central particle
    [0, 5],  # Smallest particle
    [0, -5], # Medium particle
    [5, 0]   # Largest particle
], dtype=cp.float64)

# Initialize particle colors
colors = cp.array([0, 1, 2, 3], dtype=cp.float64)

# Pre-allocate arrays for visualization
positions_host = cp.asnumpy(positions, dtype=np.float64)
velocities_host = cp.asnumpy(velocities, dtype=np.float64)
velocity_magnitudes_host = np.zeros(len(positions), dtype=np.float64)

# Create CUDA streams for concurrent execution
stream1 = cp.cuda.Stream()
stream2 = cp.cuda.Stream()

def main():
    # Set up the plot
    fig, ax = plt.subplots()
    ax.set_title(f"Body Simulation (N={len(positions)})")
    ax.set_xlim(0, SPACE_SIZE)
    ax.set_ylim(0, SPACE_SIZE)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Create scatter plot with initial empty data
    scat = ax.scatter([], [], s=2, c=[], cmap='winter')
    
    # Set particle colors
    scat.set_array(cp.asnumpy(colors))

    # Create a colorbar 
    # cbar = plt.colorbar(scat, label='Particle Velocity', aspect=40)
    # cbar.ax.set_ylabel('Particle Velocity', fontweight='bold')
    # cbar.ax.tick_params(labelsize=0)    # Remove tick labels
    # cbar.set_ticks([])                  # Remove tick marks

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
                positions, velocities, masses, sizes, 
                elasticity=ELASTICITY
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
        scat.set_array(cp.asnumpy(colors))
        
        if step == 0:
            scat.set_clim(velocity_magnitudes_host.min(), velocity_magnitudes_host.max())
        
        return scat,

    # Create video writer
    writer = FFMpegWriter(fps=FPS, metadata={'title': 'Body Simulation', 'artist': 'Matplotlib'}, bitrate=3600)

    # Save the animation
    with writer.saving(fig, OUTPUT_FILE, dpi=300):
        for step in tqdm(range(NUM_STEPS), desc="Simulating bodies"):
            update(step)
            writer.grab_frame()

    print(f"Simulation saved as {OUTPUT_FILE}")

if __name__ == "__main__": 
    main()
