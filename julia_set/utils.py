
import time
import os
import sys

# OpenCL context version, set to device
PYOPENCL_CTX_VERSION = '0'

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time for {func.__name__} is {execution_time:.4f} seconds")
        return result
    return wrapper

def avg_timing_decorator(func):
    def wrapper(*args, **kwargs):
        N = 5
        total_time = 0
        for i in range(N):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            total_time += execution_time
        avg_time = total_time / N
        print(f"Average execution time for {func.__name__} is {avg_time:.4f} seconds")
        return result
    return wrapper

def suppress_output():
    sys.stdout = open(os.devnull, 'w')

def enable_output():
    sys.stdout = sys.__stdout__
    
def check_numba_cuda():
    try:
        suppress_output()
        import numba
        from numba import cuda
        cuda.detect()
        enable_output()
        return True
    except:
        return False
    
def check_openCl():
    try:
        suppress_output()
        import pyopencl as cl
        os.environ['PYOPENCL_CTX'] = PYOPENCL_CTX_VERSION
        cl.create_some_context()
        enable_output()
        return True
    except:
        enable_output()
        return False

def check_cupy():
    try:
        # suppress_output()
        import cupy as cp
        cp.cuda.Device(0)
        # enable_output()
        return True
    except:
        enable_output()
        return False