import moderngl as mgl
import moderngl_window as mglw
from graphics import *
from sfmparser import from_openmvg, from_colmap, fetchPCD
import os
from imageio import imread_v2 as imread

def colmap_pcd_with_W2Cs(camera_meta, image_folder, plypath):
    """
    colmap offsers extrinsic in rotation and translate vector.
    so the world 2 camera should be compose of rotation in :3, :3,
    and translation in :3, 3.
    """
    keys = []
    transes = []
    rotations = []
    W2Cs = []
    for key, trans, rotmat in from_colmap(camera_meta, image_folder):
        keys.append(key)
        transes.append(trans)
        rotations.append(rotmat)

    transes = np.array(transes)
    rotations = np.array(rotations)

    for trans, rotation in zip(transes, rotations):
        W2C = np.identity(4)
        W2C[:3, :3] = rotation
        W2C[:3, 3] = trans
        W2Cs.append(W2C)

    W2Cs = np.array(W2Cs)
    pcd_points, pcd_colors = fetchPCD(plypath)
    return (pcd_points, pcd_colors, W2Cs, keys)

def openmvg_pcd_with_W2Cs(camera_meta, image_folder, plypath):
    """
    openmvg offers extrinsic in rotation and camera center, 
    so the world to camera matrix should be Rotate @ translate(-center).
    """
    def fetchPCD(path):
        from plyfile import PlyData, PlyElement
        """return positions and color"""
        plydata = PlyData.read(path)
        vertices = plydata['vertex']
        positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
        return positions, colors
        
    keys = []
    centers = []
    rotations = []
    W2Cs = []

    for key, center, rotation in from_openmvg(camera_meta, image_folder):
        if not os.path.exists(key):
            continue
        else:
            print(key)

        keys.append(key)
        centers.append(center)
        rotations.append(rotation)
    
    centers = np.array(centers)
    rotations = np.array(rotations)

    for center, rotation in zip(centers, rotations):
        R = np.identity(4)
        R[:3, :3] = rotation #.transpose() # this is transpose
        T = np.identity(4)
        T[:3, 3] = -center
        W2C = R @ T
        W2Cs.append(W2C)
    
    W2Cs = np.array(W2Cs)
    pcd_points, pcd_colors = fetchPCD(plypath)

    
    return (pcd_points, pcd_colors, W2Cs, keys)

def pcd_with_W2Cs(dataset_type, camera_meta, image_folder, plypath):
    if dataset_type == "colmap":
        return colmap_pcd_with_W2Cs(camera_meta, image_folder, plypath)
    elif dataset_type == "openmvg":
        return openmvg_pcd_with_W2Cs(camera_meta, image_folder, plypath)


class Main(Window):
    vertices = [makeCoord()]

    def __init__(self, ctx: "Context" = None, wnd: "BaseWindow" = None, timer: "BaseTimer" = None, **kwargs):
        super().__init__(ctx, wnd, timer, **kwargs)
        print("myargv:", self.myargv)

        self.fpscamera = self.camera
        self.simple_camera = Camera() 
        self.wfpscamera = bool_widget("fpscamera", False)

        self.pcd_points, self.pcd_colors, self.W2Cs, self.keys = \
            pcd_with_W2Cs(self.myargv.dataset_type, self.myargv.camera_meta, self.myargv.image_folder, self.myargv.plypath)
        
        self.wsimple_camera_index = int_widget("cam_index", 0, len(self.W2Cs)-1, 0)

        self.simple_camera._view = self.W2Cs[0]

        for w2c in self.W2Cs:
            self.vertices.append(applyMat(np.linalg.inv(w2c) @ scale(self.myargv.scale), makeCoord()))

        self.setGround()
        self.setAxis(self.vertices)
        self.setPoints(self.pcd_points, self.pcd_colors)

        for idx, key in enumerate(self.keys):
            img = imread(key)
            w2c = self.W2Cs[idx]
            c2w = np.linalg.inv(w2c)
            center = c2w[:3, 3]
            rotmat = c2w[:3, :3]
            quat = mat2quat(rotmat)
            self.setPlane(img[::-1], center=center, quat=quat, scale=[self.myargv.scale]*3)


    def xrender(self, t, frame_t):
        if self.wfpscamera():
            self.camera = self.fpscamera
        else:
            index = self.wsimple_camera_index()
            imgui.text(self.keys[index])
            self.simple_camera._view = self.W2Cs[index]
            self.camera = self.simple_camera
    
        super().xrender(t, frame_t)

if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-type", default="colmap", choices=["colmap", "openmvg"], required=True, help="sfm toolkit type")
    parser.add_argument("--camera-meta", default="db/avenue-cubic/sparse/0/images.txt", required=True, help="camera intrinsic and extrinsic file")
    parser.add_argument("--image-folder", default="db/avenue-cubic/images", required=True, help="image folder")
    parser.add_argument("--plypath", default="db/avenue-cubic/sparse/0/points3D.ply", required=True, help="point cloud ply file")
    parser.add_argument("--scale", default=0.05, type=float, help="point cloud ply file")

    argv, args = parser.parse_known_args(sys.argv[1:])
    Main.myargv = argv


    run_window_config(Main, args=args + ["--window", "pyglet"])