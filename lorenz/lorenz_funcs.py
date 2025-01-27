import numpy as np
from numba import cuda
import pyopencl as cl
import cupy as cp
import os

from utils import *

# Lorenz Attractor
# ---------------------------------------------------------------------------------------
# X' = s * (Y - X)
# Y' = r * X - Y - X * Z
# Z' = X * Y - b * Z
# s, r, b are constants
# s is the Prandtl number - controls the flow, typically 10
# r is the Rayleigh number - controls the temperature difference, typically 28
# b is the beta number - controls the geometry of the attractor, typically 8/3 ie 2.667

def lorenz_cpu(x, y, z, s, r, b, dt):
    """
    Compute the Lorenz attractor on the CPU, for a single time step.
    
    Params:
    - x, y, z: Initial values for x, y, z
    - s: Prandtl number
    - r: Rayleigh number
    - b: Beta number
    - dt: Time step size
    
    Returns:
    - x, y, z: Updated values for x, y, z
    """
    # Compute the derivatives
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    
    # Update the values, derivative * time step
    x += x_dot * dt
    y += y_dot * dt
    z += z_dot * dt
    
    return x, y, z

@cuda.jit
def lorenz_numba(x, y, z, s, r, b, dt, out_x, out_y, out_z):
    """
    Compute the Lorenz attractor on the GPU, for all time steps.
    
    Params:
    - x, y, z: Initial values for x, y, z
    - s: Prandtl number
    - r: Rayleigh number
    - b: Beta number
    - dt: Time step size
    - out_x, out_y, out_z: Output arrays for x, y, z (updated values)
    """
    # Get the index of the current thread
    i = cuda.grid(1)
    if i < x.size:
        # Compute the derivatives (index i is the current step)
        x_dot = s * (y[i] - x[i])
        y_dot = r * x[i] - y[i] - x[i] * z[i]
        z_dot = x[i] * y[i] - b * z[i]
        
        # Update the values, derivative * time step
        out_x[i] = x[i] + x_dot * dt
        out_y[i] = y[i] + y_dot * dt
        out_z[i] = z[i] + z_dot * dt


def lorenz_opencl(steps, x_init, y_init, z_init, s, r, b, dt):
    """
    Compute the Lorenz attractor on the GPU using OpenCL, for all time steps.
    
    Params:
    - steps: Number of steps to run the simulation
    - x_init, y_init, z_init: Initial values for x, y, z
    - s: Prandtl number
    - r: Rayleigh number
    - b: Beta number
    - dt: Time step size
    
    Returns:
    - out_x, out_y, out_z: Output arrays for x, y, z (updated values)
    """
    # Initialize OpenCL context and queue
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)

    # Initialize arrays to store the results, initial values for x, y, z
    x = np.full(steps, x_init).astype(np.float64)
    y = np.full(steps, y_init).astype(np.float64)
    z = np.full(steps, z_init).astype(np.float64)
    out_x = np.empty(steps).astype(np.float64)
    out_y = np.empty(steps).astype(np.float64)
    out_z = np.empty(steps).astype(np.float64)

    # Create OpenCL buffers
    mf = cl.mem_flags
    x_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=x)
    y_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=y)
    z_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=z)
    out_x_buf = cl.Buffer(context, mf.WRITE_ONLY, out_x.nbytes)
    out_y_buf = cl.Buffer(context, mf.WRITE_ONLY, out_y.nbytes)
    out_z_buf = cl.Buffer(context, mf.WRITE_ONLY, out_z.nbytes)

    # OpenCL kernel
    kernel_code = """
    __kernel void lorenz(
        __global double *x, __global double *y, __global double *z,
        const double s, const double r, const double b, const double dt,
        __global double *out_x, __global double *out_y, __global double *out_z) {
        
        int i = get_global_id(0);
        double x_val = x[i];
        double y_val = y[i];
        double z_val = z[i];
        for (int j = 0; j < i; j++) {
            double x_dot = s * (y_val - x_val);
            double y_dot = r * x_val - y_val - x_val * z_val;
            double z_dot = x_val * y_val - b * z_val;
            x_val += x_dot * dt;
            y_val += y_dot * dt;
            z_val += z_dot * dt;
        }
        out_x[i] = x_val;
        out_y[i] = y_val;
        out_z[i] = z_val;
    }
    """
    program = cl.Program(context, kernel_code).build()

    # Execute the kernel
    global_size = (steps,)
    local_size = None
    program.lorenz(queue, global_size, local_size, x_buf, y_buf, z_buf, np.float64(s), np.float64(r), np.float64(b), np.float64(dt), out_x_buf, out_y_buf, out_z_buf)

    # Copy the results from the device to the host
    cl.enqueue_copy(queue, out_x, out_x_buf).wait()
    cl.enqueue_copy(queue, out_y, out_y_buf).wait()
    cl.enqueue_copy(queue, out_z, out_z_buf).wait()

    return out_x, out_y, out_z