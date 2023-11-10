import sys; sys.path.append("./")

from graphics import *

import moderngl as mgl
from plyfile import PlyData, PlyElement
from sfmparser import from_openmvg, from_colmap
from scipy.spatial.transform import Rotation

def fetchPCD(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    return positions, colors

class ShowCamera(Window):
    def __init__(self, ctx: "Context" = None, wnd: "BaseWindow" = None, timer: "BaseTimer" = None, **kwargs):
        super().__init__(ctx, wnd, timer, **kwargs)
        self.camera.eye = np.array([0, 0.2, 0.2])
        view = self.camera.view.copy()
        self.poses = [view]

        self.setGround()
        vc = []
        self.registerAxis(vc, np.zeros(3), np.array([0, 0, 0, 1],dtype="f4"), length=5)
        self.origin_axis_xobj = self.setAxis(vc, name="origin")

        vc = []
        self.registerAxis(vc, self.camera.eye, mat2quat(np.linalg.inv(self.camera.view[:3, :3])), length=0.01)
        self.camera_axis_xobj = self.setAxis(vc, name="camera")

        # =================== COLMAP ========================
        pts, rgbs = fetchPCD("db/avenue-cubic/sparse/0/points3D.ply")
        pts *= 0.001
        self.pcd1_xobj = self.setPoints(pts, rgbs, trans=[0, 0, 0], quat=[0, 0, 0, 1], scale=np.ones(3))

        vc = []
        for key, mat in from_colmap("db/avenue-cubic/sparse/0/images.txt", "db/avenue-cubic/images").items():
            mat[:, 3] *= 0.001

            self.poses.append(mat)
            t, q = mat[:3, 3], Rotation.from_matrix(mat[:3, :3]).as_quat()
            t = t
            print("!", t, q)
            self.registerCamera(vc, t, q, length=0.001)
        self.pcd1_camera_xobj = self.setAxis(vc, "pc1")

        # =================== OPENMVG ========================
        pts, rgbs = fetchPCD("db/avenue/output/reconstruction_global/colorized.ply")
        self.pcd2_xobj = self.setPoints(pts, rgbs, trans=[0, 0, 0], quat=[0, 0, 0, 1])

        vc = []
        for key, mat in from_openmvg("db/avenue/output/reconstruction_global/sfm_data.json", "db/avenue/input").items():
            mat[:, 3] = -mat[:, 3]
            self.poses.append(mat)
            # if not os.path.exists(os.path.join("db/avenue/output/images", key)): continue
            t, q = mat[:3, 3], Rotation.from_matrix(mat[:3, :3]).as_quat()
            t = t
            print("?", t, q)
            self.registerCamera(vc, t, q, length=0.001)

            # img = imread(key)
            # img = pad_image_to_size(img, max(*img.shape[:2]), max(*img.shape[:2]))
            # self.setPlane(img, trans=t, quat=q, scale=np.ones(3)*0.001)
        self.pcd2_camera_xobj = self.setAxis(vc, name="pc2")

    def xrender(self, t, frame_t):
        super().xrender(t, frame_t)


        expanded, opened = imgui.begin("cam")
        if not hasattr(self, "pose_slider"):
            self.pose_slider = int_widget("pose_slider", 0, len(self.poses)-1, 0)

        if expanded:
            mat = self.poses[self.pose_slider()]
            imgui.text(f"{mat}\n{list(self.camera.eye), self.camera.theta, self.camera.phi}")
            self.camera.view = mat

        imgui.end()




if __name__ == "__main__":
    run_window_config(ShowCamera)