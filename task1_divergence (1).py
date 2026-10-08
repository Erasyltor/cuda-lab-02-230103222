"""Compare uniform, interleaved and warp-aligned branches on a CUDA GPU."""
import time
import numpy as np
from numba import cuda, float32

N = 1 << 20
ITERATIONS = 1000


@cuda.jit
def uniform_kernel(arr):
    idx = cuda.grid(1)
    if idx < arr.size:
        value = arr[idx]
        for _ in range(ITERATIONS):
            value = value * float32(1.0001) + float32(0.0001)
        arr[idx] = value


@cuda.jit
def interleaved_kernel(arr):
    idx = cuda.grid(1)
    if idx < arr.size:
        value = arr[idx]
        if idx % 2 == 0:
            for _ in range(ITERATIONS):
                value = value * float32(1.0001) + float32(0.0001)
        else:
            for _ in range(ITERATIONS):
                value = (value - float32(0.0001)) / float32(1.0001)
        arr[idx] = value


@cuda.jit
def warp_aligned_kernel(arr):
    idx = cuda.grid(1)
    if idx < arr.size:
        value = arr[idx]
        warp_id = idx // 32
        if warp_id % 2 == 0:
            for _ in range(ITERATIONS):
                value = value * float32(1.0001) + float32(0.0001)
        else:
            for _ in range(ITERATIONS):
                value = (value - float32(0.0001)) / float32(1.0001)
        arr[idx] = value


def benchmark():
    threads = 256
    blocks = (N + threads - 1) // threads
    h_initial = np.ones(N, dtype=np.float32)
    results = {}
    for name, kernel in (
        ("A: Uniform", uniform_kernel),
        ("B: Interleaved", interleaved_kernel),
        ("C: Warp-aligned", warp_aligned_kernel),
    ):
        d_arr = cuda.to_device(h_initial)
        kernel[blocks, threads](d_arr)  # Compile and warm up before timing.
        cuda.synchronize()
        elapsed = []
        for _ in range(10):
            # Reset outside the timer so all trials have identical input.
            d_arr.copy_to_device(h_initial)
            cuda.synchronize()
            start = time.perf_counter()
            kernel[blocks, threads](d_arr)
            cuda.synchronize()
            elapsed.append(time.perf_counter() - start)
        assert np.isfinite(d_arr.copy_to_host()).all()
        results[name] = float(np.mean(elapsed))
        print(f"{name}: {results[name]:.8f} seconds (mean of 10 trials)")
    return results


if __name__ == "__main__":
    benchmark()
