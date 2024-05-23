import numpy as np
import ctypes
from pathlib import Path


def args_cpp_nparray(array):
    args = [np.ctypeslib.ndpointer(dtype=array.dtype)]
    for i in range(0, len(array.shape)):
        args.append(ctypes.c_int)
    return args
def cpp_nparray(array):
    return array, *array.shape


class Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int), ("y", ctypes.c_int)]

class PointsPair(ctypes.Structure):
    _fields_ = [("first", ctypes.POINTER(Point)), ("first_size", ctypes.c_int), ("second", ctypes.POINTER(Point)), ("second_size", ctypes.c_int)]


def dense_match(img1:np.ndarray, img2:np.ndarray, coordinate_array:np.ndarray, epipolar_array:np.ndarray, features:int=0, window_size:int=5, lowe_ratio:float=0.8):
    lib = ctypes.CDLL(Path(__file__).parent / 'cpp/libmatching_points.so')
    lib.process_data.restype = PointsPair
    lib.process_data.argtypes = [*args_cpp_nparray(img1), *args_cpp_nparray(img2), *args_cpp_nparray(coordinate_array), *args_cpp_nparray(epipolar_array), ctypes.c_int, ctypes.c_float]
    data = lib.process_data(*cpp_nparray(img1), *cpp_nparray(img2), *cpp_nparray(coordinate_array), *cpp_nparray(epipolar_array), window_size, lowe_ratio)
    # lib.process_data.argtypes = [*args_cpp_nparray(np.expand_dims(img1_mono, axis=-1)), *args_cpp_nparray(np.expand_dims(img2_mono, axis=-1)), *args_cpp_nparray(coordinate_array), *args_cpp_nparray(epipolar), ctypes.c_int, ctypes.c_float]
    # data = lib.process_data(*cpp_nparray(np.expand_dims(img1_mono, axis=-1)), *cpp_nparray(np.expand_dims(img2_mono, axis=-1)), *cpp_nparray(coordinate_array), *cpp_nparray(epipolar), 5, 0.8)
    points_1 = []
    points_2 = []
    for i in range(data.first_size):
        points_1.append([data.first[i].x, data.first[i].y])
    for i in range(data.second_size):
        points_2.append([data.second[i].x, data.second[i].y])
    return points_1, points_2