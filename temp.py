import moderngl as mgl
from graphics import *

vertices = []
vertices.append(makeCoord())

def rotmat2mat(rotmat):
    mat = np.identity(4)
    mat[:3, :3] = rotmat
    return mat

class Temp(Window):
    def __init__(self, ctx: "Context" = None, wnd: "BaseWindow" = None, timer: "BaseTimer" = None, **kwargs):
        super().__init__(ctx, wnd, timer, **kwargs)
        self.fpscamera = self.camera
        self.simple_camera = Camera() 
        self.wfpscamera = bool_widget("fpscamera", False)

        view = np.identity(4)
        view = rotate_x(np.pi/4).transpose() @ translate(*-np.array([0, 1,1]))
        print("view matrix", view)

        self.simple_camera._view = view.astype("f4")

        R = view[:3, :3]
        t = view[:3, 3]
        C = -R.transpose() @ t # camera_center

        print("camera_center", C)

        vertices.append(applyMat(translate(*C) @ rotmat2mat(R.transpose()), makeCoord()))


        self.setGround()
        self.setAxis(vertices)

    def xrender(self, t, frame_t):
        if self.wfpscamera():
            self.camera = self.fpscamera
        else:
            self.camera = self.simple_camera
    
        super().xrender(t, frame_t)

run_window_config(Temp)