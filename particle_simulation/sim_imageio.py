import cupy as cp
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm
from datetime import datetime
import imageio.v3 as iio  # ImageIO for video writing
import imageio
import cv2

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
# SIM_LENGTH = 10        # Simulation length in seconds
NUM_STEPS = int(SIM_LENGTH * FPS)  # Number of simulation steps

# Create timestamped output filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = f"particle_simulation/vids/particle_simulation_{timestamp}.mp4"

# Initialize particle positions, velocities, and masses
positions = cp.random.uniform(0, SPACE_SIZE, size=(NUM_PARTICLES, 2))
velocities = cp.random.uniform(-1, 1, size=(NUM_PARTICLES, 2))
masses = cp.random.uniform(1, 10, size=(NUM_PARTICLES, 1))

positions = cp.asarray(positions, order='C')
velocities = cp.asarray(velocities, order='C')
masses = cp.asarray(masses, order='C')

# Pre-allocate arrays for visualization
positions_host = np.empty((NUM_PARTICLES, 2), dtype=np.float64)
velocities_host = np.empty((NUM_PARTICLES, 2), dtype=np.float64)

# Create CUDA streams for concurrent execution
stream1 = cp.cuda.Stream()
stream2 = cp.cuda.Stream()

# Initialize max velocity magnitude
max_velocity_magnitude = cp.zeros(1, dtype=cp.float64)

def main():
    # Open ImageIO writer
    writer = imageio.get_writer(OUTPUT_FILE, format="FFMPEG", fps=FPS)  # Corrected writer initialization
    scale = 800 / SPACE_SIZE


    def render_frame(step):
        global positions, velocities
        global max_velocity_magnitude

        # Physics calculations
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

        # Synchronize streams
        stream1.synchronize()
        stream2.synchronize()

        # Transfer data for rendering
        cp.asnumpy(positions, out=positions_host)
        cp.asnumpy(velocities, out=velocities_host)

        # Render particles as an image
        frame = np.zeros((800, 800, 3), dtype=np.uint8)  # Blank image for rendering
        particle_positions = (positions_host * scale).astype(int)
        velocity_magnitudes = np.linalg.norm(velocities_host, axis=1)

        if step % 10 == 0:  # Update max velocity magnitude every 10 steps
            max_velocity_magnitude = velocity_magnitudes.max()

        # Map velocity to color
        for pos, vel_mag in zip(particle_positions, velocity_magnitudes):
            color = (int(255 * (vel_mag / max_velocity_magnitude)), 0, 255 - int(255 * (vel_mag / max_velocity_magnitude)))
            pos = np.clip(pos, 0, 799)  # Ensure positions stay within frame bounds
            frame = cv2.circle(frame, tuple(pos), int(PARTICLE_RADIUS * scale), color, -1)

        # Append frame to video
        writer.append_data(frame)

    # Simulation loop
    for step in tqdm(range(NUM_STEPS), desc="Simulating particles"):
        render_frame(step)

    # Close the writer
    writer.close()
    print(f"Simulation saved as {OUTPUT_FILE}")


# if __name__ == "__main__":
#     main()

# Profiling the simulation
if __name__ == "__main__":
    import cProfile
    import pstats

    profiler = cProfile.Profile()
    profiler.enable()
    
    main()
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('cumtime')  # Sort by cumulative time
    stats.print_stats(20)  # Print the top 20 results
    stats.dump_stats("particle_simulation/prof/profile_results_imageio.prof")