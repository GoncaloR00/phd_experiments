#!/usr/bin/python3
import numpy as np

def epipolar_lines(img, F):
    """
    This function....
    img: sample image from a viewpoint
    F: Fundamental matrix
    """
    px_values = np.arange(img.shape[1])
    py_values = np.arange(img.shape[0])
    py_grid, px_grid = np.meshgrid(py_values, px_values)
    coordinate_array_N = np.dstack([px_grid, py_grid, np.ones_like(px_grid)]).reshape(-1, 3)
    epipolar = np.dot(F, coordinate_array_N.T).T
    coordinate_array = np.dstack([px_grid, py_grid]).reshape(-1, 2).astype(np.uint16)
    return epipolar, coordinate_array, coordinate_array_N








if __name__ == "__main__":
    print(f"This file must be used from other python scripts and not directly!")