import cupy as cp

cuda_code = r'''
extern "C" __global__ void handle_collisions_kernel(
    double *positions, double *velocities, double *masses, 
    int N, double particle_radius, double elasticity) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Shared memory for particle positions and velocities
    __shared__ double shared_positions[256 * 3]; // Assuming 3D positions
    __shared__ double shared_velocities[256 * 3]; // Assuming 3D velocities
    
    // Load current particle data
    double pos_i[3], vel_i[3], mass_i;
    pos_i[0] = positions[idx * 3];
    pos_i[1] = positions[idx * 3 + 1];
    pos_i[2] = positions[idx * 3 + 2];
    vel_i[0] = velocities[idx * 3];
    vel_i[1] = velocities[idx * 3 + 1];
    vel_i[2] = velocities[idx * 3 + 2];
    mass_i = masses[idx];

    for (int tile = 0; tile < (N + blockDim.x - 1) / blockDim.x; ++tile) {
        // Load shared memory for current tile
        int tile_idx = tile * blockDim.x + threadIdx.x;
        if (tile_idx < N) {
            shared_positions[threadIdx.x * 3] = positions[tile_idx * 3];
            shared_positions[threadIdx.x * 3 + 1] = positions[tile_idx * 3 + 1];
            shared_positions[threadIdx.x * 3 + 2] = positions[tile_idx * 3 + 2];
            shared_velocities[threadIdx.x * 3] = velocities[tile_idx * 3];
            shared_velocities[threadIdx.x * 3 + 1] = velocities[tile_idx * 3 + 1];
            shared_velocities[threadIdx.x * 3 + 2] = velocities[tile_idx * 3 + 2];
        }
        __syncthreads();

        // Process potential collisions within this tile
        for (int j = 0; j < blockDim.x; ++j) {
            int other_idx = tile * blockDim.x + j;
            if (other_idx >= N || idx == other_idx) continue;

            // Calculate delta position and distance
            double delta_x = pos_i[0] - shared_positions[j * 3];
            double delta_y = pos_i[1] - shared_positions[j * 3 + 1];
            double delta_z = pos_i[2] - shared_positions[j * 3 + 2];
            double distance = sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z);

            // Check for collision
            if (distance < 2 * particle_radius && distance > 0) {
                double normal_x = delta_x / distance;
                double normal_y = delta_y / distance;
                double normal_z = delta_z / distance;

                double rel_vel_x = vel_i[0] - shared_velocities[j * 3];
                double rel_vel_y = vel_i[1] - shared_velocities[j * 3 + 1];
                double rel_vel_z = vel_i[2] - shared_velocities[j * 3 + 2];

                double vel_normal = rel_vel_x * normal_x + rel_vel_y * normal_y + rel_vel_z * normal_z;

                if (vel_normal < 0) {
                    double overlap = max(2 * particle_radius - distance, 0.0);
                    double max_overlap_correction = 0.1 * particle_radius;
                    overlap = min(overlap, max_overlap_correction);

                    double factor_i = (2 * masses[other_idx]) / (mass_i + masses[other_idx]) * vel_normal;
                    double factor_j = (2 * mass_i) / (mass_i + masses[other_idx]) * vel_normal;

                    // Cap velocity changes
                    double max_velocity_change = 5;
                    double velocity_change = sqrt(factor_i * factor_i);
                    if (velocity_change > max_velocity_change) {
                        double scale = max_velocity_change / velocity_change;
                        factor_i *= scale;
                        factor_j *= scale;
                    }

                    vel_i[0] -= factor_i * normal_x;
                    vel_i[1] -= factor_i * normal_y;
                    vel_i[2] -= factor_i * normal_z;
                    shared_velocities[j * 3] += factor_j * normal_x;
                    shared_velocities[j * 3 + 1] += factor_j * normal_y;
                    shared_velocities[j * 3 + 2] += factor_j * normal_z;

                    vel_i[0] = elasticity * vel_i[0] + (1 - elasticity) * vel_i[0];
                    vel_i[1] = elasticity * vel_i[1] + (1 - elasticity) * vel_i[1];
                    vel_i[2] = elasticity * vel_i[2] + (1 - elasticity) * vel_i[2];
                }
            }

        }
        __syncthreads();
    }

    // Write back updated velocity
    velocities[idx * 3] = vel_i[0];
    velocities[idx * 3 + 1] = vel_i[1];
    velocities[idx * 3 + 2] = vel_i[2];
}
'''

# Compile the CUDA kernel
collision_kernel = cp.RawKernel(cuda_code, 'handle_collisions_kernel')

def handle_particle_collisions_cudaKernel(positions, velocities, masses, particle_radius, elasticity):
    """
    Handle particle collisions using a custom CUDA kernel with double precision.

    Args:
        positions: CuPy array of shape (N, 3) containing particle positions
        velocities: CuPy array of shape (N, 3) containing particle velocities
        masses: CuPy array of shape (N, 1) containing particle masses
        particle_radius: Radius of particles for collision detection
        elasticity: Collision elasticity coefficient
    """
    N = positions.shape[0]

    # Ensure inputs are float64 for double precision
    positions = positions.astype(cp.float64)
    velocities = velocities.astype(cp.float64)
    masses = masses.astype(cp.float64).reshape(-1)

    # Calculate grid and block dimensions
    threads_per_block = 256
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

    # Launch kernel with double precision parameters
    collision_kernel(
        (blocks_per_grid,),
        (threads_per_block,),
        (positions.ravel(), velocities.ravel(), masses, N, 
         cp.float64(particle_radius), cp.float64(elasticity))
    )

    return positions, velocities
