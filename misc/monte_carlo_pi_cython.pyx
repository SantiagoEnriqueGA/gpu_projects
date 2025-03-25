cimport cython
cimport numpy as cnp
import numpy as np  # Import Python-level numpy

cnp.import_array()
ctypedef cnp.float64_t DTYPE_t  # Use float64 for random numbers

@cython.boundscheck(False)
@cython.wraparound(False)
def _monteCarloPiCython(int num_samples):
    cdef cnp.ndarray[DTYPE_t, ndim=1] x = np.random.rand(num_samples).astype(np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] y = np.random.rand(num_samples).astype(np.float64)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] inside_circle
    cdef int count_inside = 0

    # Calculate whether points are inside the circle
    inside_circle = (x**2 + y**2 <= 1.0).astype(np.int32)
    count_inside = inside_circle.sum()

    return (count_inside / num_samples) * 4
