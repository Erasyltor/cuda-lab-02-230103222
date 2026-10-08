"""Sobel-X spatial derivative; the outer border is zero."""
import numpy as np
from numba import cuda, float32


@cuda.jit
def sobel_x_kernel(d_in, d_out, rows, cols):
    col, row = cuda.grid(2)
    if row < rows and col < cols:
        if row == 0 or row == rows - 1 or col == 0 or col == cols - 1:
            d_out[row, col] = float32(0.0)
        else:
            d_out[row, col] = (
                -d_in[row - 1, col - 1] + d_in[row - 1, col + 1]
                - float32(2.0) * d_in[row, col - 1]
                + float32(2.0) * d_in[row, col + 1]
                - d_in[row + 1, col - 1] + d_in[row + 1, col + 1]
            )


def run_sobel(h_img):
    h_img = np.ascontiguousarray(h_img, dtype=np.float32)
    if h_img.ndim != 2:
        raise ValueError("Expected a 2D image")
    rows, cols = h_img.shape
    if rows == 0 or cols == 0:
        return np.zeros_like(h_img)
    d_in = cuda.to_device(h_img)
    d_out = cuda.device_array_like(d_in)
    threads_2d = (16, 16)
    blocks_2d = ((cols + 15) // 16, (rows + 15) // 16)
    sobel_x_kernel[blocks_2d, threads_2d](d_in, d_out, rows, cols)
    return d_out.copy_to_host()


def cpu_sobel(h_img):
    out = np.zeros_like(h_img)
    out[1:-1, 1:-1] = (
        -h_img[:-2, :-2] + h_img[:-2, 2:]
        - 2.0 * h_img[1:-1, :-2] + 2.0 * h_img[1:-1, 2:]
        - h_img[2:, :-2] + h_img[2:, 2:]
    )
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    h_img = rng.random((2048, 2048), dtype=np.float32)
    result = run_sobel(h_img)
    reference = cpu_sobel(h_img)
    assert np.allclose(result, reference, atol=1e-4)
    assert np.all(result[0, :] == 0) and np.all(result[-1, :] == 0)
    assert np.all(result[:, 0] == 0) and np.all(result[:, -1] == 0)
    print(f"TASK 4 PASSED: shape={result.shape}")
    print(f"MAX DELTA = {np.max(np.abs(result - reference)):.8e}")
