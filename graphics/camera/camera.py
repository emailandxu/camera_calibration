import numpy as np
from functools import lru_cache
from graphics.utils.mathutil import projection

def getProjectionMatrix(znear=0.01, zfar=100, fovX=1.57, fovY=1.57):
    import math
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = np.zeros((4, 4))

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

class Camera():
    def __init__(self) -> None:
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
        return self._view
    
    @property
    def center(self):
        #according to https://openmvg.readthedocs.io/en/latest/openMVG/cameras/cameras/
        view = self._view
        R = view[:3, :3]
        t = view[:3, 3]
        C = -R.transpose() @ t # camera_center
        return C

    def key_event(self, key, action, modifiers):
        pass

    def mouse_drag_event(self, x, y, dx, dy):
        pass

    def mouse_scroll_event(self, x_offset, y_offset):
        pass

    def debug_gui(self, t, frame_t):
        import imgui
        imgui.text(f"camera center: {self.center}")
