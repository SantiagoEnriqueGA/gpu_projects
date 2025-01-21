import numpy as np
from numba import jit
from utils import *

@avg_timing_decorator
def fftNumpy(v):
    """Compute the Fourier Transform using NumPy."""
    return np.fft.fft(v)

@avg_timing_decorator
def fourierTransformNumpy(v):
    """Compute the Fourier Transform using NumPy."""
    N = v.shape[0]
    result = np.zeros(N, dtype=np.complex64)
    for k in range(N):
        s = 0.0
        for n in range(N):
            angle = 2j * np.pi * k * n / N
            s += v[n] * np.exp(-angle)
        result[k] = s
    return result

@avg_timing_decorator
@jit(nopython=True, parallel=True)
def fourierTransformNumba(v):
    """Compute the Fourier Transform using Numba JIT."""
    N = v.shape[0]
    result = np.zeros(N, dtype=np.complex64)
    for k in range(N):
        s = 0.0
        for n in range(N):
            angle = 2j * np.pi * k * n / N
            s += v[n] * np.exp(-angle)
        result[k] = s
    return result


def main():
    # Parameters
    N = 2_000
    
    print(f"Computing the Fourier Transform of a {N:,} element vector.")

    # Initialize a vector of random floats
    v = np.random.rand(N).astype(np.float32)
    
    v_fft = fftNumpy(v)
    v_ft_numpy = fourierTransformNumpy(v)
    v_ft_numba = fourierTransformNumba(v)

    assert np.allclose(v_fft, v_ft_numba, atol=1e-5)
    assert np.allclose(v_fft, v_ft_numpy, atol=1e-5)
    print(f"All elements of the result are equal!")
        
if __name__ == "__main__":
    main()
    
# OUTPUT:
# Computing the Fourier Transform of a 2,000 element vector.
# Average execution time for fftNumpy is 0.0000 seconds
# Average execution time for fourierTransformNumpy is 5.5902 seconds
# Average execution time for fourierTransformNumba is 0.3163 seconds
# All elements of the result are equal!