"""Three-point smoothing with replicated edge values."""
import numpy as np
from numba import cuda, float32


@cuda.jit
def stencil_1d(d_in, d_out, N):
    idx = cuda.grid(1)
    if idx < N:
        left = d_in[0] if idx == 0 else d_in[idx - 1]
        right = d_in[N - 1] if idx == N - 1 else d_in[idx + 1]
        d_out[idx] = (float32(0.25) * left + float32(0.5) * d_in[idx]
                     + float32(0.25) * right)


def run_stencil(h_in):
    h_in = np.ascontiguousarray(h_in, dtype=np.float32)
    if h_in.ndim != 1:
        raise ValueError("Expected a 1D array")
    N = h_in.size
    if N == 0:
        return h_in.copy()
    d_in = cuda.to_device(h_in)
    d_out = cuda.device_array_like(d_in)
    threads = 256
    blocks = (N + threads - 1) // threads
    stencil_1d[blocks, threads](d_in, d_out, N)
    return d_out.copy_to_host()


def cpu_stencil(arr):
    padded = np.pad(arr, (1, 1), mode="edge")
    return 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]


if __name__ == "__main__":
    N = 100_007
    h_in = np.sin(np.linspace(0, 10, N)).astype(np.float32)
    h_out_gpu = run_stencil(h_in)
    cpu_ref = cpu_stencil(h_in)
    assert np.allclose(h_out_gpu, cpu_ref, atol=1e-4)
    delta = np.max(np.abs(h_out_gpu - cpu_ref))
    print(f"TASK 2 PASSED: MAX DELTA = {delta:.8e}")
