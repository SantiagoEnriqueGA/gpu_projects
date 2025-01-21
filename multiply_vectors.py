import numpy as np
from numba import vectorize
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

def main():
    # Parameters
    N = 640_000_000
    
    print(f"Multiplying two {N:,} element vectors.")

    # Initialize two vectors of random floats
    v1 = np.random.rand(N).astype(np.float32)
    v2 = np.random.rand(N).astype(np.float32)
    
    v3 = multiplyVectors(v1, v2)
    v3_numba = multiplyVectorsNumba(v1, v2)

    assert np.allclose(v3, v3_numba)
    print(f"All elements of the result are equal!")
        
if __name__ == "__main__":
    main()


# OUTPUT:
# Multiplying two 640,000,000 element vectors.
# Average execution time for multiplyVectors is 0.8778 seconds
# Average execution time for multiplyVectorsNumba is 1.3942 seconds
# All elements of the result are equal!