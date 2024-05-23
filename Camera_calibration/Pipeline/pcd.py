#!/usr/bin/python3

import open3d as o3d
import numpy as np

import open3d as o3d

def draw_geometries_in_two_windows(geom1, geom2, title1='Window1', title2='Window2'):
    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name=title1)
    vis1.add_geometry(geom1)

    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name=title2)
    vis2.add_geometry(geom2)

    while True:
        vis1.update_geometry(geom1)
        vis1.poll_events()
        vis1.update_renderer()

        vis2.update_geometry(geom2)
        vis2.poll_events()
        vis2.update_renderer()

    vis1.destroy_window()
    vis2.destroy_window()



def remove_isolated_points(pcd, nb_neighbors=20, std_ratio=2.0):
    # Create a copy of the point cloud
    pcd_clean = pcd

    # Perform statistical outlier removal
    cl, ind = pcd_clean.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    # Return inlier point cloud
    return cl


# pcd1 = o3d.io.read_point_cloud("cloud.pcd")
# pcd2 = o3d.io.read_point_cloud("cloud2.pcd")

# # Assuming pcd1 and pcd2 are your two point clouds
# draw_geometries_in_two_windows(pcd1, pcd2, "Filtrada", "Não Filtrada")
pcd = o3d.io.read_point_cloud("cloud.pcd")
z_theshold = 0
points = np.asarray(pcd.points)
pcd = pcd.select_by_index(np.where(points[:,2] > z_theshold)[0])
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=1000, std_ratio=0.1)



# pcd = remove_isolated_points(pcd, nb_neighbors=100, std_ratio=0.1)
# pcd.cluster_dbscan(eps = 0.5, min_points=10000)
# pcd = remove_isolated_points(pcd, nb_neighbors=100, std_ratio=0.1)
# pcd = remove_isolated_points(pcd, nb_neighbors=10, std_ratio=0.1)
# print(np.asarray(pcd2.points).shape)
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.6, origin=[0, 0, 0])
o3d.visualization.draw_geometries([pcd, mesh_frame])