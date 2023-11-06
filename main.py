#!python
from functools import lru_cache
import imgui
import numpy as np

from imageio import imread

import moderngl as mgl
from moderngl_window import geometry, resources, run_window_config
from moderngl_window.opengl.vao import VAO

from graphics.base import WindowBase
from graphics.widgets import float_widget, bool_widget, float3_widget
from graphics.utils.mathutil import spherical, posemat, lookAt, projection, mat2quat, rotate_x, rotate_y, rotate_z

from util import pad_image_to_size

random_tex = lambda : (np.random.rand(255, 255, 3) * 255).astype("u1")

class FPSCamera():
    def __init__(self, speed=0.5) -> None:
        self.eye = np.array([0., 0., 1.])
        self.theta = 0.
        self.phi = np.pi/2
        self.speed = speed

    @property
    def oriental(self):
        return np.array(spherical(self.theta, np.pi/2, 1.))
    
    @property
    def look_target(self):
        return np.array(spherical(self.theta, self.phi, 1.))
    
    @property
    @lru_cache(maxsize=-1)
    def proj(self):
        return projection(fov=60)

    @property
    def view(self):
        return lookAt(eye=self.eye, at=self.eye-self.look_target, up=np.array([0, 1, 0]))

    def key_event(self, key, action, modifiers):
        if action == "ACTION_PRESS":
            if key == 119: # W
                self.eye -= self.speed * self.oriental
            elif key==97: # A
                self.eye += self.speed * np.cross(self.oriental, np.array([0.,1.,0.]))
            elif key==115: # S
                self.eye += self.speed * self.oriental
            elif key==100: # D
                self.eye -= self.speed * np.cross(self.oriental, np.array([0.,1.,0.]))
            elif key==106: # J
                self.theta+=0.1
            elif key==108: # L
                self.theta-=0.1
            elif key==105: # J
                self.phi+=0.1
            elif key==107: # K
                self.phi-=0.1
            elif key==99: # C
                self.eye[1]-= self.speed
            elif key==32: # Space
                self.eye[1]+= self.speed
            else:
                print(key)
    
    def mouse_scroll_event(self, x_offset, y_offset):
        # print(x_offset, y_offset)
        self.eye += self.oriental * -y_offset * self.speed

    def debug_gui(self):
        imgui.text(f"{self.eye.astype('f2')}")
        imgui.text(f"{np.rad2deg(self.theta):.4f}, {np.rad2deg(self.phi):.4f}")


class XObjBase():
    def __init__(self) -> None:
        self.scale = np.ones(3)
        self.trans = np.zeros(3) # x, y, z
        self.quat = np.array([0., 0., 0., 1.]) # x, y, z, w
        self.scale_offset = np.ones(3)

    @property
    def posemat(self):
        return posemat(self.trans, self.quat, self.scale * self.scale_offset)

class XObj(XObjBase):
    def __init__(self) -> None:
        super().__init__()

    def bind_vao(self, vao):
        self.vao = vao

    def bind_prog(self, prog):
        self.prog = prog

    def bind_texture(self, texture):
        self.texture = texture

    def render(self, camera, vao=None, prog=None):
        assert hasattr(camera, "view")
        assert hasattr(camera, "proj")
        
        assert vao is not None or self.vao is not None
        assert prog is not None or self.prog is not None

        if vao is not None:
            self.bind_vao(vao)

        if prog is not None:
            self.bind_prog(prog)

        self.texture.use(0)
        self.prog['texture0'].value = 0

        m, v, p = self.posemat, camera.view, camera.proj

        # note that transpose is essential, from row major to column major
        mvp = (p @ v @ m).transpose().astype("f4")
        mvp = np.ascontiguousarray(mvp) # make it contiguous
        self.prog["mvp"].write(mvp)
        # self.vao.render(self.prog)
        self.vao.render(self.prog)

class Window(WindowBase):
    def __init__(self, ctx: "mgl.Context" = None, wnd: "BaseWindow" = None, timer: "BaseTimer" = None, **kwargs):
        super().__init__(ctx, wnd, timer, **kwargs)
        self.xobjs = []
        self.xtasks = {}
        self.wfloat = float_widget("scale", 0.01, 1, 0.5)
        self.wfloat3 = float3_widget("target", 0, 3, (1, 1, 1))
        self.camera = FPSCamera()
        self.default_prog = self.load_program("default.glsl")

    def setCamera(self, trans=None, quat=None):
        x = np.array([0., 0., 0., 1., 0., 0.], dtype="f4")
        y = np.array([0., 0., 0., 0., -1., 0.], dtype="f4")
        z = np.array([0., 0., 0., 0., 0., 1.], dtype="f4")

        r = np.array([[[255, 0, 0]]], dtype="u1")
        g = np.array([[[0, 255, 0]]], dtype="u1")
        b = np.array([[[0, 0, 255]]], dtype="u1")

        def make(vertices, tex):
            xobj = XObj()

            vao = VAO(mode=mgl.LINES)
            vao.buffer(np.array(vertices, dtype="f4"), '3f', 'in_position')
            vao.buffer(np.array([0.0, 0.0, 1.0, 1.0], dtype='f4'), '2f', 'in_texcoord_0')
            texture = self.ctx.texture(tex.shape[:2], tex.shape[2], data=tex)

            xobj.bind_vao(vao)
            xobj.bind_prog(self.default_prog)
            xobj.bind_texture(texture)

            xobj.trans = trans if trans is not None else [0, 0, -3]
            xobj.quat = quat if quat is not None else [0, 0, 0, 1]
            return xobj
        
        self.xobjs.extend([
            make(x, r),
            make(y, g),
            make(z, b)
        ])
    
    def setPlane(self, tex, trans=None, quat=None):
        xobj = XObj()

        vao = geometry.quad_fs()
        texture = self.ctx.texture(tex.shape[:2], tex.shape[2], data=np.ascontiguousarray(tex[::-1]))

        xobj.bind_vao(vao)
        xobj.bind_prog(self.default_prog)
        xobj.bind_texture(texture)

        xobj.trans = trans if trans is not None else [0, 0, -3]
        xobj.quat = quat if quat is not None else [0, 0, 0, 1]
        xobj.scale = np.ones(3)

        self.xobjs.append(xobj)
    
    def key_event(self, key, action, modifiers):
        super().key_event(key, action, modifiers)
        if action=="ACTION_PRESS":
            self.xtasks[key] = lambda : self.camera.key_event(key, action, modifiers)
        else:
            self.xtasks.pop(key)

    def mouse_scroll_event(self, x_offset, y_offset):
        super().mouse_scroll_event(x_offset, y_offset)
        self.camera.mouse_scroll_event(x_offset, y_offset)
    
    def xrender(self, t, frame_t):
        imgui.text(f"{1/frame_t:.4f}")
        self.camera.debug_gui()

        scale = self.wfloat()
        self.camera.speed = scale

        assert len(self.xobjs) > 0

        for xtask in self.xtasks.values():
            xtask()

        for xobj in self.xobjs:
            # xobj.quat = mat2quat(rotate_x(t) @ rotate_y(t))
            xobj.scale_offset =  np.ones(3) * scale
            xobj.render(self.camera)

class CameraCalibrationWindow(Window):
    def __init__(self, ctx: "mgl.Context" = None, wnd: "BaseWindow" = None, timer: "BaseTimer" = None, **kwargs):
        super().__init__(ctx, wnd, timer, **kwargs)
        from camera import from_openmvg, from_colmap
        from scipy.spatial.transform import Rotation
        openmvg_path = "db/avenue/output/reconstruction_global/sfm_data.json"
        for path, mat in from_openmvg(openmvg_path).items():
        # for path, mat in from_colmap("db/avenue/output/sparse/0/images.txt", "db/avenue/output/images").items():
            mat = np.array(mat)
            trans = mat[:3, 3] * 3
            quat =  Rotation.from_matrix(mat[:3, :3]).as_quat()
            self.setCamera(trans=trans, quat=quat)
            # img = imread(path)
            # img = pad_image_to_size(img, max(*img.shape[:2]), max(*img.shape[:2]))
            # self.setPlane(img, trans=trans, quat=quat)

if __name__ == "__main__":
    run_window_config(CameraCalibrationWindow, args=["--window", "pyglet"])
