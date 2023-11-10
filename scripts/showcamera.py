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
        self.poses = [np.identity(4)]

        self.setGround()
        self.origin_axis_xobj = self.setAxis([makeCoord()], name="origin")
        self.camera_axis_xobj = self.setAxis([makeCoord()], name="camera")

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