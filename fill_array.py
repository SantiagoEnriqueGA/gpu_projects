import time
import numpy as np
from numba import jit, cuda

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
def fill_array(a, n):
    for i in range(n):
        a[i] = i

@timing_decorator
@jit
def fill_array_jit(a, n):
    for i in range(n):
        a[i] = i

N = 640_000_000
a = np.ones(N, dtype=np.float32)

fill_array(a, N)
fill_array_jit(a, N)