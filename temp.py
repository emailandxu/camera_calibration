import moderngl as mgl
from graphics import *
from sfmparser import from_openmvg, from_colmap, fetchPCD




def rotmat2mat(rotmat):
    mat = np.identity(4)
    mat[:3, :3] = rotmat
    return mat

def colmap_pcd_with_viewmat():
    viewmats = []
    for key, viewmat in from_colmap("db/avenue-cubic/sparse/0/images.txt").items():
        viewmats.append(viewmat)
    return (*fetchPCD("db/avenue-cubic/sparse/0/points3D.ply"), viewmats)

def openmvg_pcd_with_viewmat():
    pass

def pcd_with_viewmat():
    return colmap_pcd_with_viewmat()


class Temp(Window):
    vertices = [makeCoord()]

    def __init__(self, ctx: "Context" = None, wnd: "BaseWindow" = None, timer: "BaseTimer" = None, **kwargs):
        super().__init__(ctx, wnd, timer, **kwargs)
        self.fpscamera = self.camera
        self.simple_camera = Camera() 
        self.wfpscamera = bool_widget("fpscamera", False)

        self.pcd_points, self.pcd_colors, self.viewmats = pcd_with_viewmat()
        
        self.wsimple_camera_index = int_widget("cam_index", 0, len(self.viewmats)-1, 0)



        self.simple_camera._view = rotate_x(np.pi) @ self.viewmats[0]
        
        for view in self.viewmats:
            view = rotate_x(np.pi) @ view
            R = view[:3, :3]
            t = view[:3, 3]
            C = -R.transpose() @ t # camera_center
            self.vertices.append(applyMat(translate(*C) @ rotmat2mat(R.transpose()) @ scale(0.1), makeCoord()))


        self.setGround()
        self.setAxis(self.vertices)
        self.setPoints(self.pcd_points, self.pcd_colors)

    def xrender(self, t, frame_t):
        if self.wfpscamera():
            self.camera = self.fpscamera
        else:
            self.simple_camera._view = rotate_x(np.pi) @ self.viewmats[self.wsimple_camera_index()]
            self.camera = self.simple_camera
    
        super().xrender(t, frame_t)

run_window_config(Temp)