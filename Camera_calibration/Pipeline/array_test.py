#!/usr/bin/python3


import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import open3d as o3d



def triangulate_and_plot(points_hom):
    # mask = points_hom[3] != 0
    # points_hom = points_hom[:, mask]
    # points_3D = points_hom / points_hom[3]
    points_3D = points_hom
    points_3D_np = np.array(points_3D[:3, :].T, dtype=np.float64)
    # Create Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3D_np)
    
    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])
    # Plot 3D points
    # print(getsizeof(points_hom))
    # fig = plt.figure()
    
    # ax = fig.add_subplot(111, projection='3d')
    
    # ax.scatter(points_3D[0], points_3D[1], points_3D[2])
    # plt.show()



# np.set_printoptions(threshold=np.inf)
teste = np.load('teste.npy')
# print(teste)
triangulate_and_plot(teste)