#!/usr/bin/env python

import open3d as o3d
import numpy as np

# Step 2: Read the .ply file into an Open3D point cloud object
ply_file_path = "./d0_downsampling.ply"  # Replace with your file path
point_cloud = o3d.io.read_point_cloud(ply_file_path)

# Step 3: Paint all points in the point cloud (e.g., red color)
color = [1, 0, 0]  # RGB values for red
point_cloud.paint_uniform_color(color)

# Step 4: Visualize the painted point cloud
o3d.visualization.draw_geometries([point_cloud])