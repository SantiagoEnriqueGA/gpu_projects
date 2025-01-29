# GPU Projects Overview

This project contains various Python scripts and modules for performing numerical computations and simulations, focusing on GPU acceleration libraries like Numba, PyOpenCL, CuPy, and CUDA, and comparing them with CPU-based computations. The project is organized into several folders, each focusing on a specific type of computation or simulation.

## Navigation
- [Folder Overview](#folder-overview)
- [Fractal Generation](#fractal-generation)
    - [Mandelbrot Fractal](#mandelbrot-fractal)
    - [Julia Fractal](#julia-fractal)
- [Particle Simulation](#particle-simulation)
- [Lorenz Attractor](#lorenz-attractor)
- [Sorting Algorithms](#sorting-algorithms)
    - [QuickSort](#quicksort)
    - [HeapSort](#heapsort)
    - [MergeSort](#mergesort)
- [Miscellaneous Computations](#miscellaneous-computations)
    - [Fill Array](#fill-array)
    - [Fourier Transform](#fourier-transform)
    - [Monte Carlo Pi Estimation](#monte-carlo-pi-estimation)
    - [Multiply Vectors](#multiply-vectors)
    - [Prime Number Generation](#prime-number-generation)

## Folder Overview

### `mandelbrot/`
This folder contains scripts for generating Mandelbrot fractals using different computational techniques, including CPU, GPU with Numba, PyOpenCL, and CuPy.

- `mandelbrot.py`: Main script to run and time the Mandelbrot fractal on the CPU and GPU.
- `mandelbrot_funcs.py`: Functions to create the Mandelbrot fractal on the CPU and GPU.
- `benchmark_mandelbrot.py`: Script to benchmark the performance of Mandelbrot fractal generation using different techniques.
- `utils.py`: Utility functions for timing and checking the availability of computational libraries.

### `julia_set/`
This folder contains scripts for generating Julia fractals using different computational techniques, including CPU, GPU with Numba, PyOpenCL, and CuPy.

- `julia.py`: Main script to run and time the Julia fractal on the CPU and GPU.
- `julia_funcs.py`: Functions to create the Julia fractal on the CPU and GPU.
- `benchmark_julia.py`: Script to benchmark the performance of Julia fractal generation using different techniques.
- `utils.py`: Utility functions for timing and checking the availability of computational libraries.

### `particle_simulation/`
This folder contains scripts for simulating particle interactions using CUDA and other techniques.
- Main scripts:
    - `sim.py`: Main script to run the particle simulation and visualize the results. Saves the results to a file for later visualization.
    - `sim_imageio.py`: Script to run the particle simulation with imageio visualization. Visualizes the simulation in real-time.
- CUDA kernels:
    - `cuda_compute_forces.py`: CUDA kernel for computing gravitational forces.
    - `cuda_handle_boundary_collisions.py`: CUDA kernel for handling boundary collisions.
    - `cuda_handle_particle_collisions.py`: CUDA kernel for handling particle collisions.
- Utility scripts:
    - `compare_funcs.py`: Script to compare the results of different particle simulation functions. Ensures consistency across implementations.
    - `benchmark_funcs.py`: Script to benchmark the performance of various particle simulation functions.
- Profiling Results:
    - `profile_results.prof`: Profiling results for the particle simulation.
    - `profile_results_imageio.prof`: Profiling results for the particle simulation with live visualization.

### `sorting/`
This folder contains scripts for sorting algorithms and related utilities.

- `quicksort.py`: Script to perform QuickSort using different techniques, including CPU, Numba, and PyOpenCL.
- `heapsort.py`: Script to perform HeapSort using different techniques, including CPU, Numba, and PyOpenCL.
- `mergesort.py`: Script to perform MergeSort using different techniques, including CPU, Numba, and PyOpenCL.
- `utils.py`: Utility functions for timing and checking the availability of computational libraries.

### `misc/`
This folder contains miscellaneous scripts for various numerical computations and simulations. Compares the performance of different implementations using NumPy, Numba, CUDA, and PyOpenCL.

- `fill_array.py`: Script to fill an array with sequential integers using different techniques.
- `fourier_transform.py`: Script to compute the Fourier Transform using NumPy and Numba.
- `monte_carlo_pi.py`: Script to estimate Pi using the Monte Carlo method with NumPy and Numba.
- `multiply_vectors.py`: Script to multiply two vectors using NumPy and Numba.
- `prime_numbers_time.py`: Script to generate prime numbers up to a given time limit using different techniques.
- `prime_numbers_lim.py`: Script to generate prime numbers up to a given number limit using different techniques.
- `utils.py`: Utility functions for timing and checking the availability of computational libraries.


## Fractal Generation
### Mandelbrot Fractal
This project focuses on generating the Mandelbrot fractal using different computational techniques, including CPU, GPU with Numba, PyOpenCL, and CuPy. The Mandelbrot set is a set of complex numbers for which the function `f(z) = z^2 + c` does not diverge when iterated from `z = 0`. The set is defined by iterating the function until the magnitude of the result exceeds a certain threshold or the maximum number of iterations is reached. Thus the Mandelbrot set is defined as the set of complex numbers `c` for which the sequence `f(0), f(f(0)), f(f(f(0))), ...` remains bounded.

The main script `mandelbrot.py` generates the Mandelbrot fractal using different techniques and compares their performance.
The script uses the following parameters:
- `SIZE_MULTI` - Multiplier for the image size (e.g., 1 for 750x500, 2 for 1500x1000).
- `MAX_ITERS` - Maximum number of iterations for the Mandelbrot set calculation.
- `SHOW_IMAGE` - Whether to display the generated image.
- `SAVE_IMAGE` - Whether to save the generated image.
- `CPU` - Whether to run the CPU implementation. (For large images, this can be slow)

The script `mandelbrot_funcs.py` contains the functions for generating the Mandelbrot fractal on the CPU and GPU using Numba, PyOpenCL, and CuPy.

The script `benchmark_mandelbrot.py` benchmarks the performance of Mandelbrot fractal generation using different techniques and image sizes.

The script `utils.py` contains utility functions for timing and checking the availability of computational libraries.

#### Benchmark Results
The benchmark results for the Mandelbrot fractal generation as a result of running the script `benchmark_mandelbrot.py` are shown below:
```
Creating Mandelbrot Set of size 1500x1000 with 50 iterations
        Number of pixels:    1,500,000
        Pixels x Iterations: 75,000,000
------------------------------------------------------------
Running CPU version (average timing)...
Average execution time for run_cpu_avg is 8.4652 seconds

Running GPU CUDA version (average timing)...
Average execution time for run_gpu_cuda_avg is 0.1693 seconds

Running GPU OpenCL version (average timing)...
Average execution time for run_gpu_opencl_avg is 0.0723 seconds

Running GPU CuPy version (average timing)...
Average execution time for run_gpu_cupy_avg is 0.0932 seconds
```
```
Creating Mandelbrot Set of size 37500x25000 with 100 iterations
        Number of pixels:    937,500,000
        Pixels x Iterations: 93,750,000,000
------------------------------------------------------------

Running GPU CUDA version (average timing)...
Average execution time for run_gpu_cuda_avg is 4.7573 seconds

Running GPU OpenCL version (average timing)...
Average execution time for run_gpu_opencl_avg is 5.6162 seconds

Running GPU CuPy version (average timing)...
Average execution time for run_gpu_cupy_avg is 4.3302 seconds
```

#### Example Image
Here is an example of the Mandelbrot fractal generated using SIZE_MULTI=2 and MAX_ITERS=50:
![alt text](https://raw.githubusercontent.com/SantiagoEnriqueGA/gpu_projects/refs/heads/main/fractals/mandelbrot/plots/cupy_mandelbrot_size2_iters50.png)

To see a larger image, run with SIZE_MULTI=50 and MAX_ITERS=100 (37500x25000 pixels), download the image [here](https://github.com/SantiagoEnriqueGA/gpu_projects/blob/main/fractals/mandelbrot/plots/cupy_mandelbrot_size50_iters100.png).


### Julia Fractal
This project focuses on generating the Julia fractal using different computational techniques, including CPU, GPU with Numba, PyOpenCL, and CuPy. The Julia set is a set of complex numbers for which the function `f(z) = z^2 + c` does not diverge when iterated from `z = 0`. The set is defined by iterating the function until the magnitude of the result exceeds a certain threshold or the maximum number of iterations is reached. Thus the Julia set is defined as the set of complex numbers `c` for which the sequence `f(0), f(f(0)), f(f(f(0))), ...` remains bounded.

The main script `julia.py` generates the Julia fractal using different techniques and compares their performance. The script uses the following parameters:
- `SIZE_MULTI` - Multiplier for the image size (e.g., 1 for 750x500, 2 for 1500x1000).
- `MAX_ITERS` - Maximum number of iterations for the Julia set calculation.
- `SHOW_IMAGE` - Whether to display the generated image.
- `SAVE_IMAGE` - Whether to save the generated image.
- `CPU` - Whether to run the CPU implementation. (For large images, this can be slow)

The script `julia_funcs.py` contains the functions for generating the Julia fractal on the CPU and GPU using Numba, PyOpenCL, and CuPy.

The script `benchmark_julia.py` benchmarks the performance of Julia fractal generation using different techniques and image sizes.

The script `utils.py` contains utility functions for timing and checking the availability of computational libraries.

#### Benchmark Results
The benchmark results for the Julia fractal generation as a result of running the script `benchmark_julia.py` are shown below:
```
Creating Julia Set Set of size 1500x1000 with 50 iterations
        Number of pixels:    1,500,000
        Pixels x Iterations: 75,000,000
------------------------------------------------------------
Running CPU version (average timing)...
Average execution time for run_cpu_avg is 11.6283 seconds

Running GPU CUDA version (average timing)...
Average execution time for run_gpu_cuda_avg is 0.1052 seconds

Running GPU OpenCL version (average timing)...
Average execution time for run_gpu_opencl_avg is 0.0802 seconds

Running GPU CuPy version (average timing)...
Average execution time for run_gpu_cupy_avg is 0.0316 seconds
```
```
Creating Julia Set Set of size 37500x25000 with 100 iterations
        Number of pixels:    937,500,000
        Pixels x Iterations: 93,750,000,000
------------------------------------------------------------

Running GPU CUDA version (average timing)...
Average execution time for run_gpu_cuda_avg is 4.5497 seconds

Running GPU OpenCL version (average timing)...
Average execution time for run_gpu_opencl_avg is 5.5448 seconds

Running GPU CuPy version (average timing)...
Average execution time for run_gpu_cupy_avg is 4.2457 seconds
```


#### Example Image
Here is an example of the Mandelbrot fractal generated using SIZE_MULTI=2 and MAX_ITERS=50:
![alt text](https://raw.githubusercontent.com/SantiagoEnriqueGA/gpu_projects/refs/heads/main/fractals/julia_set/plots/cupy_julia_size2_iters50.png)

To see a larger image, run with SIZE_MULTI=50 and MAX_ITERS=100 (37500x25000 pixels), download the image [here](https://github.com/SantiagoEnriqueGA/gpu_projects/blob/main/fractals/mandelbrot/plots/cupy_mandelbrot_size50_iters100.png).

## Particle Simulation
This project focuses on simulating particle interactions using CUDA and other techniques. The simulation involves particles initialized with random positions, velocities, and masses, interacting through gravitational forces and colliding with each other and the boundaries of the simulation space. The particles are colored based on their velocity to visualize their movement. Particle positions, velocities, and masses are all calculated as float64 arrays for high precision.

### Main Scripts
`sim.py`: Run the particle simulation and visualize the results. Each frame of the simulation is saved with matplotlib's FFmpeg writer for visualization.  
`sim_imageio.py`: Run the particle simulation with live visualization. The simulation is visualized in real-time using the ImageIO library for video writing.

**The following are the high level steps of the simulation:**
1. Set constants for the simulation (number of particles, simulation space size, etc.).
2. Initialize particle positions, velocities, and masses.
3. Run the simulation loop:
    - Compute gravitational forces between particles.
    - Handle boundary collisions.
    - Handle particle collisions.
    - Update particle positions and velocities.
    - Visualize the particles (Real-time or save frames).

**The following steps were performed to optimize the simulation:**
1. Implement CUDA kernels for computing gravitational forces, handling boundary collisions, and handling particle collisions.
    - CUDA kernels are used to parallelize the computation on the GPU.
2. Utilized pinned memory for faster data transfer between the CPU and GPU.
    - Pinned memory allows for faster data transfer by avoiding the overhead of page-locked memory.
3. Pre-allocated memory for particle data to avoid memory allocation overhead during the simulation.
4. Created CUDA streams for asynchronous memory transfers and kernel execution.
    - CUDA streams are used to overlap memory transfers and kernel execution for better performance.
5. Profiled the simulation to identify bottlenecks and optimize the code.
    - Profiling helps identify areas of the code that can be optimized for better performance.
    - The profiling results are saved in `profile_results.prof`.
    - Profiling shows that after optimization, the simulation spends most of its time converting to Numpy arrays for visualization and writing frames to disk.

### Example Video
The following video shows the simulation running for 30 seconds with the default parameters:
![video](https://raw.githubusercontent.com/SantiagoEnriqueGA/gpu_projects/refs/heads/main/particle_simulation/vids/particle_simulation_ex_30secs.mp4)

### CUDA Kernels
Each of the following CUDA kernels is implemented in a separate file using CuPy's RawKernel interface which allows for direct CUDA kernel execution by passing the kernel code written in CUDA C.  
- `cuda_compute_forces.py`: CUDA kernel for computing gravitational forces between particles.  
    - Takes particle positions, velocities, masses, the gravitational constant, and epsilon (softening factor) as input.
    - Returns the forces acting on each particle.
- `cuda_handle_boundary_collisions.py`: CUDA kernel for handling boundary collisions.  
    - Takes particle positions, velocities, the simulation space size, and the coefficient of elasticity as input.
    - Returns the updated particle positions and velocities after collisions.
- `cuda_handle_particle_collisions.py`: CUDA kernel for handling particle collisions.  
    - Takes particle positions, velocities, masses, radii, and the coefficient elasticity as input.
    - Returns the updated particle positions and velocities after collisions.

### Utility Scripts
- `compare_funcs.py`: Compare the results of different particle simulation functions to ensure consistency across implementations.
- `benchmark_funcs.py`: Benchmark the performance of various particle simulation functions to compare their speed.
- `utils.py`: Utility functions for timing and checking the availability of computational libraries.

### Benchmark Results
The benchmark results for the particle simulation as a result of running the script `benchmark_funcs.py` are shown below:
```
Benchmarking 5,000 particles in a 200.00 x 200.00 space with 10 runs each.
----------------------------------------------------------------------------------------------------

Benchmarking gravitational force computations...
compute_forces_np average time:                         0.87466 seconds (std_dev: 0.03086 seconds)
compute_forces_cp average time:                         0.87244 seconds (std_dev: 0.00243 seconds)
compute_forces_cudaKernel average time:                 0.02130 seconds (std_dev: 0.00090 seconds)
----------------------------------------------------------------------------------------------------
Speedup:                                                41.06x

Benchmarking particle-particle collision handling...
handle_particle_collisions_np average time:             1.40059 seconds (std_dev: 0.57684 seconds)
handle_particle_collisions_cp average time:             1.17647 seconds (std_dev: 0.00940 seconds)
handle_particle_collisions_cKDTree average time:        0.80803 seconds (std_dev: 1.10441 seconds)
handle_particle_collisions_cudaKernel average time:     0.05238 seconds (std_dev: 0.02090 seconds)
----------------------------------------------------------------------------------------------------
Speedup:                                                26.74x

handle_boundary_collisions average time:                0.04352 seconds (std_dev: 0.12656 seconds)
handle_boundary_collisions_cudaKernel average time:     0.00027 seconds (std_dev: 0.00056 seconds)

Benchmarking CuPy specific functions...
compute_forces_cudaKernel:              CPU:   221.151 us   +/- 52.04`5 (min:   132.900 / max:   358.000) us    GPU-0: 76154.183 us   +/- 3094.332 (min: 69212.318 / max: 83601.852) us
handle_particle_collisions_cudaKernel:  CPU:   157.586 us   +/- 41.083 (min:   112.800 / max:   307.400) us     GPU-0: 24926.697 us   +/- 1309.562 (min: 22241.920 / max: 26700.640) us
handle_boundary_collisions_cudaKernel:  CPU:    82.937 us   +/-  9.473 (min:    76.500 / max:   125.100) us     GPU-0:    98.740 us   +/- 12.590 (min:    85.984 / max:   143.104) us
```

## Lorenz Attractor
This project focuses on simulating the Lorenz Attractor using different computational techniques. This is a classic example of a chaotic system that exhibits sensitive dependence on initial conditions. It is also a problem that is not easily parallelizable due to the iterative nature of the calculations, making it a good candidate for comparison between CPU and GPU implementations.
The Lorenz Attractor is a set of chaotic solutions to a system of differential equations that exhibit sensitive dependence on initial conditions. The attractor is defined by the following equations:
$$
\begin{aligned}
X' &= \sigma (Y - X) \\
Y' &= X (\rho - Z) - Y \\
Z' &= X Y - \beta Z
\end{aligned}
$$
where $X, Y, Z$ are the state variables, and $\sigma, \rho, \beta$ are the parameters of the system.

### Main Scripts
- `lorenz.py`: Main script to run the Lorenz Attractor simulation using different techniques and compare their performance.
- `lorenz_funcs.py`: Functions to calculate the Lorenz Attractor using different techniques.
- `benchmark_lorenz.py`: Script to benchmark the performance of the Lorenz Attractor simulation.

#### Lorenz Attractor Output
![alt text](https://raw.githubusercontent.com/SantiagoEnriqueGA/gpu_projects/refs/heads/main/lorenz/lorenz.png)


#### Benchmark Results
The benchmark results for the Lorenz Attractor simulation as a result of running the script `benchmark_lorenz.py` are shown below. Note that the GPU version times increase significantly with the number of steps, indicating that the GPU is not as efficient as the CPU at this particular problem.
```
Running 1000 steps of Lorenz attractor... 
        Running CPU version...         execution time: 0.0010 seconds
        Running GPU CUDA version...   execution time: 1.4790 seconds
Running 5000 steps of Lorenz attractor...
        Running CPU version...         execution time: 0.0030 seconds
        Running GPU CUDA version...   execution time: 4.6690 seconds
Running 10000 steps of Lorenz attractor...
        Running CPU version...         execution time: 0.0060 seconds
        Running GPU CUDA version...   execution time: 11.0260 seconds
Running 15000 steps of Lorenz attractor...
        Running CPU version...         execution time: 0.0090 seconds
        Running GPU CUDA version...   execution time: 19.6380 seconds
Running 20000 steps of Lorenz attractor...
        Running CPU version...         execution time: 0.0110 seconds
        Running GPU CUDA version...   execution time: 25.2714 seconds
Running 25000 steps of Lorenz attractor...
        Running CPU version...         execution time: 0.0150 seconds
        Running GPU CUDA version...   execution time: 32.9117 seconds
Running 30000 steps of Lorenz attractor... 
        Running CPU version...         execution time: 0.0180 seconds
        Running GPU CUDA version...   execution time: 40.4746 seconds
Running 35000 steps of Lorenz attractor... 
        Running CPU version...         execution time: 0.0200 seconds
        Running GPU CUDA version...   execution time: 53.2662 seconds
Running 40000 steps of Lorenz attractor... 
        Running CPU version...         execution time: 0.0230 seconds
        Running GPU CUDA version...   execution time: 64.6176 seconds
```
![alt text](https://raw.githubusercontent.com/SantiagoEnriqueGA/gpu_projects/refs/heads/main/lorenz/bench_lorenz.png)



## Sorting Algorithms
This project focuses on sorting algorithms and their implementations using different techniques, including CPU, Numba, and PyOpenCL. The sorting algorithms are implemented for sorting arrays of integers. Currently, the project includes the QuickSort algorithm. Other sorting algorithms can be added in the future.

### QuickSort
The QuickSort algorithm is a comparison-based sorting algorithm that divides the input array into two subarrays based on a pivot element. The subarrays are then recursively sorted. The QuickSort algorithm has an average time complexity of O(n log n) and is widely used in practice due to its efficiency. In the worst-case scenario, the algorithm has a time complexity of O(n^2).

The script `quicksort.py` contains the implementation of the QuickSort algorithm. Each implementation is timed to compare the performance compared to the standard NumPy implementation. Each implementation is tested with the same input array and compared for correctness. The following implementations are included:
- `NumPy`: Standard NumPy implementation of QuickSort.
- `CPU`: Base CPU implementation using Python.
- `Numba`: Numba implementation of QuickSort for JIT compilation.
- `PyOpenCL`: PyOpenCL implementation of QuickSort for GPU acceleration.

```
Sorting array of size 1,000,000 elements.
------------------------------------------------------------
Execution time for sort_numpy is 0.0550 seconds
Execution time for quicksort_cpu is 4.9470 seconds
Execution time for quicksort_numba is 1.3436 seconds
Execution time for quicksort_opencl is 5.1661 seconds
All elements of the result are equal!
```

Performance benchmarks are conducted to compare the speed of each implementation for sorting large arrays of integers. The results are plotted to visualize the performance differences between the implementations:
![alt text](https://raw.githubusercontent.com/SantiagoEnriqueGA/gpu_projects/refs/heads/main/sorting/quicksort_comparison.png)

### HeapSort
The HeapSort algorithm is a comparison-based sorting algorithm that uses a binary heap data structure to sort elements. The algorithm has a time complexity of O(n log n) and is an in-place sorting algorithm. In the worst-case scenario, the algorithm has a space complexity of O(1).

The script `heapsort.py` contains the implementation of the HeapSort algorithm using different techniques, including CPU, Numba, and PyOpenCL. The performance of each implementation is compared for sorting large arrays of integers. The following implementations are included:
- `NumPy`: Standard NumPy implementation of HeapSort.
- `CPU`: Base CPU implementation using Python.
- `Numba`: Numba implementation of HeapSort for JIT compilation.
- `PyOpenCL`: PyOpenCL implementation of HeapSort for GPU acceleration.

```
Sorting array of size 1,000,000 elements.
------------------------------------------------------------
Execution time for sort_numpy is 0.1383 seconds
Execution time for heapsort_cpu is 11.7589 seconds
Execution time for heapsort_numba is 1.3883 seconds
Execution time for heapsort_opencl is 10.3213 seconds
All elements of the result are equal!
```

Performance benchmarks are conducted to compare the speed of each implementation for sorting large arrays of integers. The results are plotted to visualize the performance differences between the implementations:
![alt text](https://raw.githubusercontent.com/SantiagoEnriqueGA/gpu_projects/refs/heads/main/sorting/heapsort_comparison.png)


### MergeSort
The MergeSort algorithm is a comparison-based sorting algorithm that divides the input array into two halves, recursively sorts the halves, and then merges them. The algorithm has a time complexity of O(n log n) and is stable and efficient for large datasets. In the worst-case scenario, the algorithm has a space complexity of O(n). The script `mergesort.py` contains the implementation of the MergeSort algorithm using different techniques, including CPU, Numba, and PyOpenCL. The performance of each implementation is compared for sorting large arrays of integers. The following implementations are included:
- `NumPy`: Standard NumPy implementation of MergeSort.
- `CPU`: Base CPU implementation using Python.
- `Numba`: Numba implementation of MergeSort for JIT compilation.
- `PyOpenCL`: PyOpenCL implementation of MergeSort for GPU acceleration.

```
Sorting array of size 1,000,000 elements.
------------------------------------------------------------
Execution time for sort_numpy is 0.0705 seconds
Execution time for mergesort_cpu is 8.0944 seconds
Execution time for mergesort_numba is 1.3210 seconds
Execution time for mergesort_opencl is 1.7379 seconds
All elements of the result are equal!
```

Performance benchmarks are conducted to compare the speed of each implementation for sorting large arrays of integers. The results are plotted to visualize the performance differences between the implementations:
![alt text](https://raw.githubusercontent.com/SantiagoEnriqueGA/gpu_projects/refs/heads/main/sorting/mergesort_comparison.png)


## Miscellaneous Computations
This folder contains miscellaneous scripts for various numerical computations and simulations. The scripts compare the performance of different implementations using NumPy, Numba, CUDA, and PyOpenCL.

### Fill Array
The script `fill_array.py` fills an array with sequential integers using different techniques, including NumPy and Numba. The performance of each implementation is compared for large array sizes.

An example of the performance comparison is shown below:
```
Filling an array of 64,000,000 elements.
Average execution time for fill_array is 7.3506 seconds
Average execution time for fill_array_numba is 0.1083 seconds
All elements of the result are equal!
```

### Fourier Transform
The script `fourier_transform.py` computes the Fourier Transform of a signal using NumPy custom and FFT implementations and Numba. The performance of each implementation is compared for different signal sizes.

An example of the performance comparison is shown below:
```
Computing the Fourier Transform of a 2,000 element vector.
Average execution time for fftNumpy is 0.0000 seconds
Average execution time for fourierTransformNumpy is 5.5902 seconds
Average execution time for fourierTransformNumba is 0.3163 seconds
All elements of the result are equal!
```

### Monte Carlo Pi Estimation
The script `monte_carlo_pi.py` estimates the value of Pi using the Monte Carlo method with NumPy and Numba. The performance of each implementation is compared for different numbers of samples. A visualization of the Monte Carlo simulation for multiple sample sizes is also provided.

An example of the performance comparison is shown below:
```
Estimating Pi using 100,000,000 samples.
Average execution time for monteCarloPiNumpy_AVG is 2.1093 seconds
Average execution time for monteCarloPiNumba_AVG is 1.9931 seconds
Pi estimate using NumPy: 3.14150216
Pi estimate using Numba: 3.14172632

Estimating Pi using 10,000 samples.
        Average execution time for NumPy: 0.0002 seconds
        Average execution time for Numba: 0.1284 seconds
Estimating Pi using 100,000 samples.
        Average execution time for NumPy: 0.0029 seconds
        Average execution time for Numba: 0.0011 seconds
Estimating Pi using 1,000,000 samples.
        Average execution time for NumPy: 0.0205 seconds
        Average execution time for Numba: 0.0089 seconds
Estimating Pi using 10,000,000 samples.
        Average execution time for NumPy: 0.2071 seconds
        Average execution time for Numba: 0.0950 seconds
Estimating Pi using 100,000,000 samples.
        Average execution time for NumPy: 2.0562 seconds
        Average execution time for Numba: 1.0004 seconds
```

**Visualizing the time taken to estimate Pi using the Monte Carlo method with different sample sizes:**
![alt text](https://raw.githubusercontent.com/SantiagoEnriqueGA/gpu_projects/refs/heads/main/misc/monte_carlo_pi.png)


### Multiply Vectors
The script `multiply_vectors.py` multiplies two vectors using NumPy and Numba (JIT and Vectorize) implementations. The performance of each implementation is compared for large vector sizes.

An example of the performance comparison is shown below:
```
Multiplying two 640,000,000 element vectors.
Average execution time for multiplyVectors is 1.9930 seconds
Average execution time for multiplyVectorsNumba is 4.1995 seconds
Average execution time for multiplyVectorsNumbaJit is 3.0660 seconds
All elements of the result are equal!
```

### Prime Number Generation
The scripts `prime_numbers_time.py` and `prime_numbers_lim.py` generate prime numbers up to a given time or number limit using different techniques, including CPU, Numba, PyOpenCL, and CuPy. The performance of each implementation is compared with each other to ensure correctness and speed.

An example of the performance comparison based on time is shown below:
```
Generating prime numbers within 30 seconds.
------------------------------------------------------------
CPU    produced 274,449    prime numbers in 30 seconds
Numba  produced 1,513,139  prime numbers in 30 seconds
OpenCL produced 5,978,516  prime numbers in 30 seconds
CuPy   produced 7,378,188  prime numbers in 30 seconds
All methods produced the same results!
```

An example of the performance comparison based on the number limit is shown below:
```
Generating prime numbers up to 10,000,000.
------------------------------------------------------------
Execution time for primes_cpu is 1.8090 seconds
Execution time for primes_numba is 0.8160 seconds
Execution time for primes_opencl is 1.1751 seconds
Execution time for primes_cupy is 0.5790 seconds
All methods produced the same results!
Last 5 prime numbers: [9999937, 9999943, 9999971, 9999973, 9999991]
```

#### Visualization of Prime Number Generation Functions
![alt text](https://raw.githubusercontent.com/SantiagoEnriqueGA/gpu_projects/refs/heads/main/misc/prime_numbers_lim.png)
![alt text](https://raw.githubusercontent.com/SantiagoEnriqueGA/gpu_projects/refs/heads/main/misc/prime_numbers_time.png)
