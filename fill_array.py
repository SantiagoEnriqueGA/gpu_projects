import numpy as np
from numba import jit, cuda
from utils import *

@avg_timing_decorator
def fill_array(a, n):
    """Fill an array with sequential integers."""
    for i in range(n):
        a[i] = i

@avg_timing_decorator
@jit
def fill_array_jit(a, n):
    """Fill an array with sequential integers using Numba's @jit decorator."""
    for i in range(n):
        a[i] = i


def main():
    # Parameters
    N = 64_000_000

    # Initialize an array of ones
    a = np.ones(N, dtype=np.float32)

    # Run the fill_array function
    fill_array(a, N)
    print(f"Last 5 elements of the result: {a[-5:]}")
    print("")
    fill_array_jit(a, N)
    print(f"Last 5 elements of the result: {a[-5:]}")
    
if __name__ == "__main__":
    main()    


# OUTPUT:
# Average execution time for fill_array is 7.1856 seconds
# Last 5 elements of the result: [63999996. 63999996. 63999996. 64000000. 64000000.]

# Average execution time for fill_array_jit is 0.1723 seconds
# Last 5 elements of the result: [63999996. 63999996. 63999996. 64000000. 64000000.]