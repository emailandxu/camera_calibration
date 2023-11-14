import numpy as np
from functools import lru_cache
from graphics.utils.mathutil import projection, rotate_x
from .base import CameraBase

def getProjectionMatrix(znear=0.01, zfar=100, fovX=1.57, fovY=1.57):
    import math
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = np.zeros((4, 4))


    # opengl camera is oppsite to sfm toolkit
    z_sign = -1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

class Camera(CameraBase):
    def __init__(self) -> None:
        self.scroll_factor = 0.

        self._view = np.identity(4)
        self._view[:3, 2] = np.array([0, 0, -1])
        self._view[:3, 3] = -np.array([0, 1, 2])
        self._view = self._view.astype("f4")

    @property
    @lru_cache(maxsize=-1)
    def proj(self):
        # return projection(fov=45, near=0.001)
        return getProjectionMatrix()

    @property
    def view(self):
        center = self.center
        center += self.scroll_factor * self._view[2, :3]

        t = self.trans_from_center(center)
        # rotate x 180 becasue the two sfm toolkit assume y axis pointing to bottom
        return rotate_x(np.pi) @ np.array([
            [ *self._view[0, :3], t[0]],
            [ *self._view[1, :3], t[1]],
            [ *self._view[2, :3], t[2]],
            [ *self._view[3]]
        ])
    
    @property
    def center(self):
        #according to https://openmvg.readthedocs.io/en/latest/openMVG/cameras/cameras/
        view = self._view
        R = view[:3, :3]
        t = view[:3, 3]
        C = -R.transpose() @ t # camera_center
        return C
    
    def trans_from_center(self, center):
        R = np.identity(4)
        R[:3, :3] = self._view[:3, :3]
        t = np.identity(4)
        t[:3, 3] = center
        return -(R @ t)[:3, 3]

    def key_event(self, key, action, modifiers):
        pass

    def mouse_drag_event(self, x, y, dx, dy):
        pass

    def mouse_scroll_event(self, x_offset, y_offset):
        self.scroll_factor += 0.1 * y_offset

    def debug_gui(self, t, frame_t):
        import imgui
        imgui.text(f"camera center: {self.center}")
        imgui.text(f"scroll factor: {self.scroll_factor:.1f}")
