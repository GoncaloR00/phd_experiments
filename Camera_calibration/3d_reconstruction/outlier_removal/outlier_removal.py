import numpy as np

def outlier_removal(pcd, nb_neighbors=1000, std_ratio=0.1):
    points = np.asarray(pcd.points)
    # Remove points with coord z<0
    pcd_out = pcd.select_by_index(np.where(points[:,2] > 0)[0])
    pcd_out, _ = pcd_out.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd_out