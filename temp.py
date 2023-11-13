import moderngl as mgl
from graphics import *
from sfmparser import from_openmvg, from_colmap, fetchPCD
import os


def rotmat2mat(rotmat):
    mat = np.identity(4)
    mat[:3, :3] = rotmat
    return mat

def colmap_pcd_with_W2Cs():
    keys = []
    transes = []
    rotations = []
    W2Cs = []
    for key, trans, rotmat in from_colmap("db/avenue-cubic/sparse/0/images.txt"):
        keys.append(key)
        transes.append(trans)
        rotations.append(rotmat)

    transes = np.array(transes)
    rotations = np.array(rotations)

    for trans, rotation in zip(transes, rotations):
        R = np.identity(4)
        R[:3, :3] = rotation
        T = np.identity(4)
        T[:3, 3] = trans
        W2C = R @ T
        W2Cs.append(W2C)

    W2Cs = np.array(W2Cs)
    pcd_points, pcd_colors = fetchPCD("db/avenue-cubic/sparse/0/points3D.ply")
    return (pcd_points, pcd_colors, W2Cs)

def openmvg_pcd_with_W2Cs():
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
    orientaions = []
    W2Cs = []

    for key, center, orientaion in from_openmvg("db/avenue/output/sfm_data_perspective.json", "db/avenue/output/images"):
        if not os.path.exists(key):
            continue
        else:
            print(key)

        keys.append(key)
        centers.append(center)
        orientaions.append(orientaion)
    
    centers = np.array(centers)
    orientaions = np.array(orientaions)

    for center, orientaion in zip(centers, orientaions):
        R = np.identity(4)
        R[:3, :3] = orientaion.transpose()
        T = np.identity(4)
        T[:3, 3] = -center
        W2C = R @ T
        W2Cs.append(W2C)
    
    W2Cs = np.array(W2Cs)
    pcd_points, pcd_colors = fetchPCD("db/avenue/output/reconstruction_global/colorized.ply")

    
    return (pcd_points, pcd_colors, W2Cs)

def pcd_with_W2Cs():
    return openmvg_pcd_with_W2Cs()


class Temp(Window):
    vertices = [makeCoord()]

    def __init__(self, ctx: "Context" = None, wnd: "BaseWindow" = None, timer: "BaseTimer" = None, **kwargs):
        super().__init__(ctx, wnd, timer, **kwargs)
        self.fpscamera = self.camera
        self.simple_camera = Camera() 
        self.wfpscamera = bool_widget("fpscamera", False)

        self.pcd_points, self.pcd_colors, self.W2Cs = pcd_with_W2Cs()
        
        self.wsimple_camera_index = int_widget("cam_index", 0, len(self.W2Cs)-1, 0)

        self.simple_camera._view = self.W2Cs[0]

        for w2c in self.W2Cs:
            self.vertices.append(applyMat(np.linalg.inv(w2c) @ scale(0.01), makeCoord()))


        self.setGround()
        self.setAxis(self.vertices)
        self.setPoints(self.pcd_points, self.pcd_colors)

    def xrender(self, t, frame_t):
        if self.wfpscamera():
            self.camera = self.fpscamera
        else:
            self.simple_camera._view = self.W2Cs[self.wsimple_camera_index()]
            self.camera = self.simple_camera
    
        super().xrender(t, frame_t)

run_window_config(Temp)