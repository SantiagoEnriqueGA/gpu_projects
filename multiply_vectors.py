import time
import numpy as np
from numba import vectorize

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time for {func.__name__} is {execution_time} seconds")
        return result
    return wrapper

@timing_decorator
def multiplyVectors(v1, v2):
    return v1 * v2

@timing_decorator
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

    print(f"\n")

    v3 = multiplyVectorsNumba(v1, v2)
    print(f"Last 5 elements of the result: {v3[-5:]}")

        
if __name__ == "__main__":
    main()