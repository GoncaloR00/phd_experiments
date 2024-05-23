import cv2
import numpy as np
import open3d as o3d

def triangulate(P1, P2, points1, points2, img1, img2):
    # Triangulate points
    points_hom = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
    points_3D = points_hom / points_hom[3]
    points_3D_np = np.array(points_3D[:3, :].T, dtype=np.float64)

    # Convert points to integer for indexing
    points1 = points1.astype(int)
    points2 = points2.astype(int)

    # Interpolate colors
    colors1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)[points1[:, 1], points1[:, 0]]
    colors2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)[points2[:, 1], points2[:, 0]]
    # colors = ((colors1 + colors2) / 2).astype(np.uint8)
    colors = colors1.astype(np.uint8)

    # Create Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3D_np)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255)
    
    return pcd


# def triangulate(P1, P2, points1, points2, img1, img2):
#     # Triangulate points
#     points_hom = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
#     points_3D = points_hom / points_hom[3]
#     points_3D_np = np.array(points_3D[:3, :].T, dtype=np.float64)

#     # Convert points to integer for indexing
#     points1 = points1.astype(int)
#     points2 = points2.astype(int)

#     # Interpolate colors
#     colors1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)[points1[:, 1], points1[:, 0]]
#     colors2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)[points2[:, 1], points2[:, 0]]
#     colors = colors1.astype(np.uint8)

#     # Filter points that are not within the FOV of both cameras
#     mask1 = (points1[:, 0] >= 0) & (points1[:, 0] < img1.shape[1]) & (points1[:, 1] >= 0) & (points1[:, 1] < img1.shape[0])
#     mask2 = (points2[:, 0] >= 0) & (points2[:, 0] < img2.shape[1]) & (points2[:, 1] >= 0) & (points2[:, 1] < img2.shape[0])
#     mask = mask1 & mask2

#     points_3D_np = points_3D_np[mask]
#     colors = colors[mask]

#     # Create Open3D PointCloud object
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points_3D_np)
#     pcd.colors = o3d.utility.Vector3dVector(colors / 255)
    
#     return pcd