#!/Users/marky/Desktop/BVC/dmr_project/dmr_python/env/bin/python

from pathlib import Path
import numpy as np
import csv
import re
import cv2
import open3d as o3d
import sys

def read_calib(calib_file_path):

    with open(calib_file_path, 'r') as calib_file:
        calib = {}
        csv_reader = csv.reader(calib_file, delimiter='=')
        for attr, value in csv_reader:
            calib.setdefault(attr, value)

    return calib

def read_pfm(pfm_file_path):

    with open(pfm_file_path, 'rb') as pfm_file:
        header = pfm_file.readline().decode().rstrip()
        channels = 3 if header == 'PF' else 1

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', pfm_file.readline().decode('utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception("Malformed PFM header.")

        scale = float(pfm_file.readline().decode().rstrip())
        if scale < 0:
            endian = '<' # little endian
            scale = -scale
        else:
            endian = '>' # big endian

        disparity = np.fromfile(pfm_file, endian + 'f')

    img = np.reshape(disparity, newshape=(height, width, channels))
    img = np.flipud(img).astype('uint8')
        
    # show(img, "disparity")

    return disparity, [(height, width, channels), scale]

def create_depth_map(pfm_file_path, calib=None):

    disparity, [shape,scale] = read_pfm(pfm_file_path)

    if calib is None:
        raise Exception("Loss calibration information.")
    else:
        fx = float(calib['cam0'].split(' ')[0].lstrip('['))
        # fy = float(calib['cam0'].split(' ')[5].rstrip(';'))
        base_line = float(calib['baseline'])
        doffs = float(calib['doffs'])

        # scale factor is used here
        depth_map = fx*base_line / (disparity / scale + doffs)
        depth_map = np.reshape(depth_map, newshape=shape)
        depth_map = np.flipud(depth_map).astype('uint8')
        # print(disparity.shape)

        dmap = DepthMap(shape[0], shape[1], depth_map)
        return dmap

def show(img, win_name='image'):

    if img is None:
        raise Exception("Can't display an empty image.")
    else:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.imshow(win_name, img)
        cv2.waitKey()
        cv2.destroyWindow(win_name)

def intrinsic_from_fov(height, width, fov=90):
    """
    Basic Pinhole Camera Model
    intrinsic params from fov and sensor width and height in pixels
    """
    px, py = (width / 2, height / 2)
    hfov = fov / 360. * 2. * np.pi
    fx = width / (2. * np.tan(hfov / 2.))

    vfov = 2. * np.arctan(np.tan(hfov / 2) * height / width)
    fy = height / (2. * np.tan(vfov / 2.))

    return np.array([[fx, 0, px, 0.],
                     [0, fy, py, 0.],
                     [0, 0, 1., 0.],
                     [0., 0., 0., 1.]])

def main(filepath):

    pfm_file_dir = Path(r'dataset/', filepath)
    calib_file_path = pfm_file_dir.joinpath('calib.txt')
    disp_left = pfm_file_dir.joinpath('disp0.pfm')

    # calibration information
    calib = read_calib(calib_file_path)
    # create depth map
    depth_map_left = create_depth_map(disp_left, calib)
    # print(depth_map_left)
    # show(depth_map_left.map, "depth_map")

    height = depth_map_left.height
    width = depth_map_left.width
    K = intrinsic_from_fov(height, width)
    u0 = K[0, 2]
    v0 = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]

    cam_points = np.zeros((height * width, 3))
    i = 0
    for v in range(height):
        for u in range(width):
            x = (u - u0) * depth_map_left.dmap[v, u] / fx
            y = (v - v0) * depth_map_left.dmap[v, u] / fy
            z = depth_map_left.dmap[v, u]
            cam_points[i] = (x, y, z)
            i += 1
    cam_points = cam_points.T

    pcd_cam = o3d.geometry.PointCloud()
    pcd_cam.points = o3d.utility.Vector3dVector(cam_points.T[:, :3])
    pcd_cam.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcd_cam])

class DepthMap:
    def __init__(self, height, width, dmap):
        self.height = height
        self.width = width
        self.dmap = dmap

if __name__ == '__main__':

    #weird error if I try to error check for number of arguments?! rip
    main(sys.argv[1]) 

