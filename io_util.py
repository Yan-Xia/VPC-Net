import numpy as np
from open3d import *


def read_pcd(filename):
    pcd = open3d.io.read_point_cloud(filename)
    return np.array(pcd.points)


def save_pcd(filename, points):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    open3d.io.write_point_cloud(filename, pcd)
