#!/Users/marky/Desktop/BVC/dmr_project/dmr_python/env/bin/python

from pathlib import Path
import numpy as np
import csv
import re
import cv2
import open3d as o3d
import sys
import time

#reads calibration file from the middlebury dataset
def read_calib(calib_file_path):

    with open(calib_file_path, 'r') as calib_file:
        calib = {}
        csv_reader = csv.reader(calib_file, delimiter='=')
        for attr, value in csv_reader:
            calib.setdefault(attr, value)

    return calib

#reads a pfm image from the middlebury dataset
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

#displays the image
def show(img, win_name='image'):

    if img is None:
        raise Exception("Can't display an empty image.")
    else:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.imshow(win_name, img)
        cv2.waitKey()
        cv2.destroyWindow(win_name)


#Basic Pinhole Camera Model
#intrinsic params from fov and sensor width and height in pixels
def intrinsic_from_fov(height, width, fov=90):
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
    disp_right = pfm_file_dir.joinpath('disp1.pfm')
    img_left = pfm_file_dir.joinpath('im0.png')
    img_right = pfm_file_dir.joinpath('im1.png')

    colors_left = cv2.imread(str(img_left)) #bgr #for some reason it doesn't work if I don't do str()
    colors_left = cv2.cvtColor(colors_left, cv2.COLOR_BGR2RGB) #rgb
    size = int(colors_left.size/3)
    colors_left = colors_left.flatten()
    colors_left = [x/255 for x in colors_left]
    colors_left = np.reshape(colors_left, (size, 3))
    print(colors_left)

    # calibration information
    calib = read_calib(calib_file_path)

    # create depth maps (L, R)
    depth_map_left = create_depth_map(disp_left, calib)
    depth_map_right = create_depth_map(disp_right, calib)

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
    pcd_cam.colors = o3d.utility.Vector3dVector(colors_left) #use image colors

    pcd_cam.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]) #to prevent it from being upside down
    o3d.visualization.draw_geometries([pcd_cam])

def pixel_coord_np(width, height):
    """
    Pixel in homogenous coordinate
    Returns:
        Pixel coordinate:       [3, width * height]
    """
    x = np.linspace(0, width - 1, width).astype(np.int)
    y = np.linspace(0, height - 1, height).astype(np.int)
    [x, y] = np.meshgrid(x, y)
    return np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))

class DepthMap:
    def __init__(self, height, width, dmap):
        self.height = height
        self.width = width
        self.dmap = dmap

if __name__ == '__main__':

    #weird error when I try to error check for number of arguments?! rip
    main(sys.argv[1]) 

