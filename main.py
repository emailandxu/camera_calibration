#!python
from functools import lru_cache
import imgui
import numpy as np

from imageio import imread

import moderngl as mgl
from moderngl_window import geometry, resources, run_window_config
from moderngl_window.opengl.vao import VAO

from graphics.base import WindowBase
from graphics.widgets import float_widget, bool_widget
from graphics.utils.mathutil import spherical, posemat, lookAt, projection, mat2quat, rotate_x, rotate_y, rotate_z

random_tex = lambda : (np.random.rand(255, 255, 3) * 255).astype("u1")

class FPSCamera():
    def __init__(self) -> None:
        self.eye = np.array([0., 0., 1.])
        self.theta = 0.
        self.phi = np.pi/2

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
                self.eye -= self.oriental
            elif key==97: # A
                self.eye += np.cross(self.oriental, np.array([0.,1.,0.]))
            elif key==115: # S
                self.eye += self.oriental
            elif key==100: # D
                self.eye -= np.cross(self.oriental, np.array([0.,1.,0.]))
            elif key==106: # J
                self.theta+=0.1
            elif key==108: # L
                self.theta-=0.1
            elif key==105: # J
                self.phi+=0.1
            elif key==107: # K
                self.phi-=0.1
            elif key==99: # C
                self.eye[1]-= 1
            elif key==32: # Space
                self.eye[1]+= 1
            else:
                print(key)

    def debug_gui(self):
        imgui.text(f"{self.eye.astype('f2')}")
        imgui.text(f"{self.theta:.4f}, {self.phi:.4f}")


class XObjBase():
    def __init__(self) -> None:
        self.scale = np.ones(3)
        self.trans = np.zeros(3) # x, y, z
        self.quat = np.array([0., 0., 0., 1.]) # x, y, z, w

    @property
    def posemat(self):
        return posemat(self.trans, self.quat, self.scale)

class XObj(XObjBase):
    def __init__(self, tex:"np.ndarray"=None) -> None:
        super().__init__()
        
        if tex is None:
            self.tex = (np.ones((1, 1, 3)) * 255 ).astype("u1")
        else:
            assert tex.dtype == np.uint8 
            assert len(tex.shape) == 3
            self.tex = tex

    @property
    def texture_size(self):
        return self.tex.shape[:2]

    @property
    def texture_channel(self):
        return self.tex.shape[2]

    def bind_vao(self, vao):
        self.vao = vao

    def bind_prog(self, prog):
        self.prog = prog

    def bind_texture(self, texture):
        self.texture = texture
        self.texture.write(self.tex)

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

        # note that transpose is essential, don't know why.
        mvp = (p @ v @ m).transpose().astype("f4")
        mvp = np.ascontiguousarray(mvp) # make it contiguous
        self.prog["mvp"].write(mvp)
        self.vao.render(self.prog)

class Window(WindowBase):
    def __init__(self, ctx: "mgl.Context" = None, wnd: "BaseWindow" = None, timer: "BaseTimer" = None, **kwargs):
        super().__init__(ctx, wnd, timer, **kwargs)
        self.xobjs = []
        self.xinit()
        self.camera = FPSCamera()

    def xinit(self):
        tex = imread("resources/spot/spot_texture.png")
        tex = np.ascontiguousarray(tex.transpose(1, 0, 2))
        print(tex.shape, tex.dtype)
        xobj = self.create(np.ascontiguousarray(tex[:,::-1]))
        xobj.trans = [-1, 0, -3]
        xobj2 = self.create(tex)
        xobj2.trans = [0, 0, -3]

        self.xobjs.append(xobj)
        self.xobjs.append(xobj2)

    def create(self, tex, trans=None, quat=None):
        xobj = XObj(tex)
        # vao = self.load_scene("eqrec/eqrec.obj").root_nodes[0].mesh.vao
        vao = self.load_scene("spot/spot.obj").root_nodes[0].mesh.vao
        # vao = geometry.cube()
        xobj.bind_vao(vao)
        xobj.bind_prog(self.load_program("default.glsl"))
        xobj.bind_texture(self.ctx.texture(xobj.texture_size, xobj.texture_channel))

        xobj.trans = trans if trans is not None else [0, 0, -3]
        xobj.quat = quat if quat is not None else mat2quat(rotate_y(np.pi/2) @ rotate_z(np.pi * 3/2))
        return xobj
    
    def key_event(self, key, action, modifiers):
        super().key_event(key, action, modifiers)
        self.camera.key_event(key, action, modifiers)

    def xrender(self, t, frame_t):
        imgui.text(f"{1/frame_t:.4f}")
        self.camera.debug_gui()

        for xobj in self.xobjs:
            xobj.quat = mat2quat(rotate_x(t) @ rotate_y(t))
            xobj.render(self.camera)

if __name__ == "__main__":
    run_window_config(Window, args=["--window", "pyglet"])
