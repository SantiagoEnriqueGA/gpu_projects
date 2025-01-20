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

    # Initialize two vectors of random floats
    v1 = np.random.rand(N).astype(np.float32)
    v2 = np.random.rand(N).astype(np.float32)
    
    v3 = multiplyVectors(v1, v2)
    print(f"Last 5 elements of the result: {v3[-5:]}")
    print("")
    v3 = multiplyVectorsNumba(v1, v2)
    print(f"Last 5 elements of the result: {v3[-5:]}")

        
if __name__ == "__main__":
    main()


# OUTPUT:
# Average execution time for multiplyVectors is 0.9624 seconds
# Last 5 elements of the result: [0.05277812 0.3918012  0.18786642 0.40295723 0.5274713 ]

# Average execution time for multiplyVectorsNumba is 3.0426 seconds
# Last 5 elements of the result: [0.05277812 0.3918012  0.18786642 0.40295723 0.5274713 ]