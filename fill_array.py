import time
import numpy as np
from numba import jit, cuda
from utils import timing_decorator, avg_timing_decorator

@avg_timing_decorator
def fill_array(a, n):
    for i in range(n):
        a[i] = i

@avg_timing_decorator
@jit
def fill_array_jit(a, n):
    for i in range(n):
        a[i] = i


def main():
    N = 64_000_000
    a = np.ones(N, dtype=np.float32)

    fill_array(a, N)
    print(f"Last 5 elements of the result: {a[-5:]}")
    
    print("")
    
    fill_array_jit(a, N)
    print(f"Last 5 elements of the result: {a[-5:]}")
    
if __name__ == "__main__":
    main()    