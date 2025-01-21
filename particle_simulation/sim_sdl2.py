import cupy as cp
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm
from datetime import datetime

import sdl2
import sdl2.ext

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

SIM_LENGTH = 3        # Simulation length in seconds
NUM_STEPS = int(SIM_LENGTH * FPS)  # Number of simulation steps

# Create timestamped output filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = f"particle_simulation/vids/particle_simulation_{timestamp}.mp4"

# Initialize particle positions, velocities, and masses
positions = cp.random.uniform(0, SPACE_SIZE, size=(NUM_PARTICLES, 2))
velocities = cp.random.uniform(-1, 1, size=(NUM_PARTICLES, 2))
masses = cp.random.uniform(1, 10, size=(NUM_PARTICLES, 1))

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
    # Initialize SDL2
    sdl2.ext.init()

    # Create window with the correct flags
    window = sdl2.ext.Window(
        "Particle Simulation", 
        size=(800, 800),
        flags=sdl2.SDL_WINDOW_SHOWN | sdl2.SDL_WINDOW_OPENGL
    )        
    # Create renderer with hardware acceleration
    renderer = sdl2.ext.Renderer(
        window,
        flags=sdl2.SDL_RENDERER_ACCELERATED | sdl2.SDL_RENDERER_PRESENTVSYNC
    )
    # Set blend mode for transparency
    renderer.blendmode = sdl2.SDL_BLENDMODE_BLEND

    def update(step):
        global positions, velocities
        
        # Physics calculations (same as before)
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
        
        # Clear screen
        renderer.clear(sdl2.ext.Color(0, 0, 0))
        
        # Get positions and velocities from GPU
        pos = cp.asnumpy(positions)
        vel = cp.asnumpy(velocities)
        vel_mag = np.linalg.norm(vel, axis=1)
        vel_normalized = (vel_mag - vel_mag.min()) / (vel_mag.max() - vel_mag.min())
        
        # Draw particles
        for i in range(NUM_PARTICLES):
            x, y = pos[i]
            # Map position from simulation space to screen space
            screen_x = int(x * 800/SPACE_SIZE)
            screen_y = int(y * 800/SPACE_SIZE)
            
            # Color based on velocity (blue to white gradient)
            v = vel_normalized[i]
            color = sdl2.ext.Color(
                int(255 * v),     # R
                int(255 * v),     # G
                255               # B
            )
            
            # Draw particle
            renderer.draw_point((screen_x, screen_y), color)
        
        # Present the rendered frame
        renderer.present()
    
    # Main simulation loop
    running = True
    step = 0

    with tqdm(total=NUM_STEPS, desc="Simulating particles") as pbar:
        while running and step < NUM_STEPS:
            events = sdl2.ext.get_events()
            for event in events:
                if event.type == sdl2.SDL_QUIT:
                    running = False
                    break

            update(step)
            step += 1
            pbar.update(1)
            # sdl2.SDL_Delay(1000 // FPS) 

    # Cleanup
    renderer.destroy()
    sdl2.ext.quit()

if __name__ == "__main__":
    main()
