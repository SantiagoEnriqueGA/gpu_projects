
import time

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time for {func.__name__} is {execution_time} seconds")
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
        print(f"Average execution time for {func.__name__} is {avg_time} seconds")
        return result
    return wrapper