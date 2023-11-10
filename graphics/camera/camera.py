import numpy as np
from functools import lru_cache
from graphics.utils.mathutil import projection

class Camera():
    def __init__(self) -> None:
        self._view = np.identity(4)
        self._view[:3, 2] = np.array([0, 0, -1])
        self._view[:3, 3] = -np.array([0, 1, 2])
        self._view = self._view.astype("f4")

    @property
    @lru_cache(maxsize=-1)
    def proj(self):
        return projection(fov=45, near=0.001)

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
