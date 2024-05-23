import importlib
import numpy as np

current_package = '.'.join(__name__.split('.')[:-1])

def feature_match(image_a:np.ndarray, image_b: np.ndarray, algorithm:str="sift_py",  features:int = None, lowe_ratio:float = None, visualization:bool = 0):
    module_name = current_package + '.' + algorithm + '.' + algorithm
    module = importlib.import_module(module_name)
    params = {
    'features': features,
    'lowe_ratio': lowe_ratio,
    }
    filtered_params = {k: v for k, v in params.items() if v is not None}
    return module.feature_match(image_a, image_b, **filtered_params, visualization = visualization)
