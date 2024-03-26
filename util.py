import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import math
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as Rotation
import pymap3d as pm
import laspy
from scipy.spatial import KDTree

def plotpointcloud(data, colors=None, fname=None, show_axis=False):
    '''
        plot pointcloud based on open3d

        Parameters:
            data: np.array
                [3, n] or [n, 3] array representing the point cloud.
            colors: np.array or None
                [n, 3] array representing the colors of the point cloud.
                None representing no color
            fname: str or None
                A string representing the name of the plot to be saved on the disk.
                None representing not saving
            show_axis: bool
                True representing plot axis; False otherwise
    '''
    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    if(data.shape[0]==3):
        pc.points = o3d.utility.Vector3dVector(data.T)
    else:
        pc.points = o3d.utility.Vector3dVector(data)
    if(type(colors) != type(None)):
        pc.colors = o3d.utility.Vector3dVector(colors)

    if(type(fname) != type(None)):
        o3d.io.write_point_cloud(fname, pc)

    if(show_axis):
        viewer = o3d.visualization.Visualizer()
        viewer.create_window()
        viewer.add_geometry(pc)
        opt = viewer.get_render_option()
        opt.show_coordinate_frame = True
        viewer.run()
        viewer.destroy_window()
    else:
        o3d.visualization.draw_geometries([pc])


def plotpointcloud_plt(data, colors=None):
    '''
        plot pointcloud based on matplotlib
        Only used if open3d can not be installed. It can not plot when the number of points is large

        Parameters:
            data: np.array
                [3, n] or [n, 3] array representing the point cloud.
            colors: np.array or None
                [n, 3] array representing the colors of the point cloud.
                None representing no color
    '''
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    if(type(colors) == type(None)):
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=1)
    else:
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=1, c=colors)
    min_data = data.min(axis=0)
    max_data = data.max(axis=0)
    scale_data = (max_data - min_data).max()
    middle_data = (min_data + max_data) / 2
    ax.set_xlim3d(middle_data[0] - scale_data, middle_data[0] + scale_data)
    ax.set_ylim3d(middle_data[1] - scale_data, middle_data[1] + scale_data)
    ax.set_zlim3d(middle_data[2] - scale_data, middle_data[2] + scale_data)
    plt.show()


def read_las_file(fname):
    '''
        read *.las file

        Parameters:
            fname: str
                A string representing the *.las file to be read
        Return:
            np.array
            [n, 3] array representing the point cloud
    '''
    with laspy.open(fname) as fh:
        las = fh.read()
        dataset = np.vstack([las.X, las.Y, las.Z]).transpose()
    return dataset


def backproject_depth(depth, inv_K, mask=False):
    '''
        back project the depth image to generate a point cloud

        Parameters:
            depth: np.array
                [h, w] array representing the depth image.
            inv_K: np.array
                [3, 3] array representing the inverse of the intrinsic matrix
            mask: bool
                True representing removing points with invalid depth value.
        Return:
            np.array
            [n, 3] array representing the point cloud
    '''
    h, w = depth.shape
    idu, idv = np.meshgrid(range(w), range(h))
    grid = np.stack((idu.flatten(), idv.flatten(), np.ones([w * h])))
    x = np.matmul(inv_K[:3, :3], grid)
    x = x * depth.flatten()[None, :]
    x = x.T
    #print(np.min(depth))
    if mask:
        #x = x[(depth.flatten() > 0)]
        x = x[(depth.flatten() < 1000)]
    return x


def transform4x4(pc, T):
    '''
        apply rigid transformation on a point cloud

        Parameters:
            pc: np.array
                [n, 3] array representing the point cloud.
            T: np.array
                [4, 4] array representing the rigid transformation matrix

        Return:
            np.array
                [n, 3] array representing the point cloud after the transformation

    '''
    return (np.matmul(T[:3, :3], pc.T) + T[:3, 3:4]).T


def extract_local_norm(pc):
    '''
            compute single point normal

            Parameters:
                pc: np.array
                    [n, 3] array representing the neighbor point cloud.

            Return:
                np.array
                    [3] array representing the point normal
    '''
    c = pc.mean(0)
    CV = (pc-c).T@(pc-c)
    w, v = np.linalg.eigh(CV)
    return v[:,0]


def extract_all_local_norm(pc, k = 10):
    '''
            compute point normal of all points

            Parameters:
                pc: np.array
                    [n, 3] array representing the neighbor point cloud.
                k: int
                    an integer representing the size of the neighbor points

            Return:
                np.array
                    [n, 3] array representing the point normal of all points
    '''
    tree = KDTree(pc)
    _, idx = tree.query(pc, k)
    normals = np.zeros((len(pc), 3))
    for i in range(len(pc)):
        normals[i, :] = extract_local_norm(pc[idx[i]])
    return normals


def extract_local_normwithplanescore(pc):
    '''
            compute single point normal together with its plane score

            Parameters:
                pc: np.array
                    [n, 3] array representing the neighbor point cloud.

            Return:
                np.array
                    [3] array representing the point normal
    '''
    c = pc.mean(0)
    CV = (pc-c).T@(pc-c)
    w, v = np.linalg.eigh(CV)
    return v[:,0], w[0]/(w[0]+w[1]+w[2])
