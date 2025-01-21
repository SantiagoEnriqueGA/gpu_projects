import numpy as np
from numba import jit, cuda
from utils import *

@avg_timing_decorator
def fill_array(a, n):
    """Fill an array with sequential integers."""
    for i in range(n):
        a[i] = i
    return a


@avg_timing_decorator
@jit
def fill_array_numba(a, n):
    """Fill an array with sequential integers using Numba's @jit decorator."""
    for i in range(n):
        a[i] = i
    return a

def main():
    # Parameters
    N = 64_000_000

    print(f"Filling an array of {N:,} elements.")

    # Initialize an array of ones
    a = np.ones(N, dtype=np.float32)

    # Run the fill_array function
    a = fill_array(a, N)
    a_numba = fill_array_numba(a, N)
    
    assert np.allclose(a, a_numba)
    print(f"All elements of the result are equal!")
    
    
    
if __name__ == "__main__":
    main()    


# OUTPUT:
# Filling an array of 64,000,000 elements.
# Average execution time for fill_array is 7.3506 seconds
# Average execution time for fill_array_numba is 0.1083 seconds
# All elements of the result are equal!