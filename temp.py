import moderngl as mgl
from graphics import *

vertices = []
vertices.append(applyMat(rotate_z(np.pi/4) @ rotate_x(np.pi/4) @ translate(0, 1, 0), makeCoord()))
vertices.append(applyMat(rotate_x(np.pi/4) @ translate(0, 1, 0), makeCoord()))

class Temp(Window):
    def __init__(self, ctx: "Context" = None, wnd: "BaseWindow" = None, timer: "BaseTimer" = None, **kwargs):
        super().__init__(ctx, wnd, timer, **kwargs)
        self.camera.dragable = True
        self.camera.speed = 1
        self.setGround()
        self.setAxis(vertices)

run_window_config(Temp)