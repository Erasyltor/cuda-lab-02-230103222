"""Scale arbitrary-length vectors using exactly 64 blocks of 256 threads."""
import numpy as np
from numba import cuda


@cuda.jit
def grid_stride_scale_kernel(d_arr, factor, N):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, N, stride):
        d_arr[i] = d_arr[i] * factor


def run_grid_stride(h_arr, factor):
    h_arr = np.ascontiguousarray(h_arr, dtype=np.float32)
    if h_arr.ndim != 1:
        raise ValueError("Expected a 1D array")
    if h_arr.size == 0:
        return h_arr.copy()
    d_arr = cuda.to_device(h_arr)
    grid_stride_scale_kernel[64, 256](d_arr, np.float32(factor), h_arr.size)
    return d_arr.copy_to_host()


if __name__ == "__main__":
    N = 1 << 24
    factor = np.float32(4.25)
    h_arr = np.ones(N, dtype=np.float32)
    result = run_grid_stride(h_arr, factor)
    assert np.allclose(result, factor)
    print(f"TASK 3 PASSED: all {N:,} elements equal {factor}")
    print("Launch: 64 blocks x 256 threads = 16,384 threads")
