import time
import numpy as np
from numba import vectorize
from utils import timing_decorator, avg_timing_decorator

@avg_timing_decorator
def multiplyVectors(v1, v2):
    return v1 * v2

@avg_timing_decorator
@vectorize(['float32(float32, float32)'], target='parallel') # target='cuda' for GPU
def multiplyVectorsNumba(v1, v2):
    return v1 * v2


def main():
    N = 640_000_000
    # N = 1_000_000

    v1 = np.random.rand(N).astype(np.float32)
    v2 = np.random.rand(N).astype(np.float32)
    
    v3 = multiplyVectors(v1, v2)
    print(f"Last 5 elements of the result: {v3[-5:]}")

    print("")

    v3 = multiplyVectorsNumba(v1, v2)
    print(f"Last 5 elements of the result: {v3[-5:]}")

        
if __name__ == "__main__":
    main()