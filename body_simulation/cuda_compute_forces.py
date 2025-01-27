import cupy as cp

# CUDA kernel for computing gravitational forces with double precision
cuda_code = r'''
extern "C" __global__
void compute_forces_kernel(const double* positions,
                         const double* masses,
                         double* forces,
                         const int N,
                         const double G,
                         const double epsilon) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < N) {
        double fx = 0.0;
        double fy = 0.0;
        double pos_i_x = positions[i * 2];
        double pos_i_y = positions[i * 2 + 1];
        double mass_i = masses[i];
        
        // Compute forces from all other particles
        for (int j = 0; j < N; j++) {
            if (i != j) {
                double dx = -(positions[j * 2] - pos_i_x);
                double dy = -(positions[j * 2 + 1] - pos_i_y);
                
                double dist_squared = dx * dx + dy * dy + epsilon;
                double dist = sqrt(dist_squared);
                
                double force = G * mass_i * masses[j] / dist_squared;
                
                fx += force * dx / dist;
                fy += force * dy / dist;
            }
        }
        
        forces[i * 2] = fx;
        forces[i * 2 + 1] = fy;
    }
}
'''

# Compile the CUDA kernel
force_kernel = cp.RawKernel(cuda_code, 'compute_forces_kernel')

def compute_forces_cudaKernel(positions, masses, G=1.0, epsilon=1e-5):
    """
    Compute gravitational forces using a custom CUDA kernel with double precision.
    
    Args:
        positions: CuPy array of shape (N, 2) containing particle positions
        masses: CuPy array of shape (N, 1) containing particle masses
        G: Gravitational constant
        epsilon: Small value to avoid division by zero
    
    Returns:
        CuPy array of shape (N, 2) containing the forces on each particle
    """
    N = positions.shape[0]
    
    # Ensure inputs are float64 for double precision
    positions = positions.astype(cp.float64)
    masses = masses.astype(cp.float64).reshape(-1)
    
    # Allocate output array with double precision
    forces = cp.zeros((N, 2), dtype=cp.float64)
    
    # Calculate grid and block dimensions
    threads_per_block = 256
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block
    
    # Launch kernel with double precision parameters
    force_kernel(
        (blocks_per_grid,),
        (threads_per_block,),
        (positions, masses, forces, N, cp.float64(G), cp.float64(epsilon))
    )
    
    return forces