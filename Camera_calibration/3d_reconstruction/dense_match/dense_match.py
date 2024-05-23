import importlib
import numpy as np

current_package = '.'.join(__name__.split('.')[:-1])


def dense_match(img1:np.ndarray, img2:np.ndarray, coordinate_array:np.ndarray, epipolar_array:np.ndarray, algorithm:str="norm_l1", features:int=None, window_size:int=None, lowe_ratio:float=None):
    module_name = current_package + '.' + algorithm + '.' + algorithm
    module = importlib.import_module(module_name)
    params = {
    'features': features,
    'window_size': window_size,
    'lowe_ratio': lowe_ratio,
    }
    filtered_params = {k: v for k, v in params.items() if v is not None}
    return module.dense_match(img1, img2, coordinate_array, epipolar_array, **filtered_params)