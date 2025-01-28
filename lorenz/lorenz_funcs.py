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
    Compute the Lorenz attractor on the GPU, computing sequential time steps.
    Each thread computes one time step based on the previous step's results.
    
    Params:
    - x, y, z: Arrays containing values for x, y, z
    - s: Prandtl number
    - r: Rayleigh number
    - b: Beta number
    - dt: Time step size
    - out_x, out_y, out_z: Output arrays for x, y, z (updated values)
    """
    i = cuda.grid(1)
    
    # Skip if thread index is out of bounds
    if i >= x.size - 1:  # -1 because we need to access i+1
        return
        
    # Compute the derivatives using current values
    x_dot = s * (y[i] - x[i])
    y_dot = r * x[i] - y[i] - x[i] * z[i]
    z_dot = x[i] * y[i] - b * z[i]
    
    # Update the next position in the arrays
    out_x[i + 1] = x[i] + x_dot * dt
    out_y[i + 1] = y[i] + y_dot * dt
    out_z[i + 1] = z[i] + z_dot * dt
    
    # First element remains unchanged (initial condition)
    if i == 0:
        out_x[0] = x[0]
        out_y[0] = y[0]
        out_z[0] = z[0]

