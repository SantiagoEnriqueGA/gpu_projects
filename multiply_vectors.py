import numpy as np
from numba import vectorize, jit
from utils import *

@avg_timing_decorator
def multiplyVectors(v1, v2):
    """Multiply two vectors using NumPy."""
    return v1 * v2

@avg_timing_decorator
@vectorize(['float32(float32, float32)'], target='parallel') # target='cuda' for GPU
def multiplyVectorsNumba(v1, v2):
    """Multiply two vectors using Numba Vectorized Functions."""
    return v1 * v2

@avg_timing_decorator
@jit(nopython=True, parallel=True)
def multiplyVectorsNumbaJit(v1, v2):
    """Multiply two vectors using Numba JIT."""
    return v1 * v2

def main():
    # Parameters
    N = 640_000_000
    
    print(f"Multiplying two {N:,} element vectors.")

    # Initialize two vectors of random floats
    v1 = np.random.rand(N).astype(np.float32)
    v2 = np.random.rand(N).astype(np.float32)
    
    v3 = multiplyVectors(v1, v2)
    v3_numba = multiplyVectorsNumba(v1, v2)
    v3_numba_jit = multiplyVectorsNumbaJit(v1, v2)

    assert np.allclose(v3, v3_numba)
    assert np.allclose(v3, v3_numba_jit)
    print(f"All elements of the result are equal!")
        
if __name__ == "__main__":
    main()


# OUTPUT:
# Multiplying two 640,000,000 element vectors.
# Average execution time for multiplyVectors is 1.9930 seconds
# Average execution time for multiplyVectorsNumba is 4.1995 seconds
# Average execution time for multiplyVectorsNumbaJit is 3.0660 seconds
# All elements of the result are equal!