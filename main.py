#!/home/xushuli/miniconda3/envs/meshdiff/bin/python
import moderngl as mgl
from moderngl_window import geometry, resources
from moderngl_window.opengl.vao import VAO
import numpy as np

import imgui
from graphics.base import WindowBase
from graphics.widgets import float_widget, bool_widget
from graphics.utils.mathutil import spherical, posemat, lookAt, projection, mat2quat, rotate_x, rotate_y

class XObjBase():
    def __init__(self) -> None:
        self.scale = np.ones(3)
        self.trans = np.zeros(3) # x, y, z
        self.quat = np.array([0., 0., 0., 1.]) # x, y, z, w

    @property
    def posemat(self):
        return posemat(self.trans, self.quat, self.scale)


class XObj(XObjBase):
    def __init__(self, vao:"VAO", window:"Window", tex:"np.ndarray"=None) -> None:
        super().__init__()
        
        self.vao = vao
        self.window = window
        self.prog = window.load_program("default.glsl")
        
        if tex is None:
            self.tex = (np.ones((1, 1, 3)) * 255 ).astype("u1")
        else:
            assert tex.dtype == np.uint8 
            assert len(tex.shape) == 3
            self.tex = tex
    
        size, components = self.tex.shape[:2], self.tex.shape[-1]
        self.texture : mgl.Texture = window.ctx.texture(size, components)

    def render(self):
        self.texture.write(self.tex)
        self.texture.use(0)
        self.prog['texture0'].value = 0

        m, v, p = self.posemat, self.window.view, self.window.proj

        # note that transpose is essential, don't know why.
        mvp = (p @ v @ m).transpose().astype("f4").copy()
        
        self.prog["mvp"].write(mvp)
        self.vao.render(self.prog)


class Window(WindowBase):
    def __init__(self, ctx: "mgl.Context" = None, wnd: "BaseWindow" = None, timer: "BaseTimer" = None, **kwargs):
        super().__init__(ctx, wnd, timer, **kwargs)
        self.wtheta = float_widget("theta", -np.pi, np.pi, 0.)
        self.wphi = float_widget("phi", 0, np.pi, np.pi/2)
        self.wdist = float_widget("distance", 0., 2., 1.0)
        self.wrot = bool_widget("rotate", True)

        self.proj = projection(fov=60)
        self.view = np.identity(4)
        self.xobjs = []
        self.xinit()
 
    def xinit(self):
        scene = self.load_scene("eqrec/eqrec.obj")
        vao = scene.root_nodes[0].mesh.vao
        xobj = XObj(vao, self)
        xobj.trans = [0, 0, -3]
        self.xobjs.append(xobj)
    
    def xrender(self, t, frame_t):
        self.eye = spherical(self.wtheta(), self.wphi(), self.wdist())
        self.view = lookAt(eye=self.eye, at=np.array([0,0,0]), up=np.array([0, 1, 0]))
        
        for xobj in self.xobjs:
            xobj.quat = mat2quat(rotate_x(t) @ rotate_y(t))
            xobj.render()

if __name__ == "__main__":
    Window.run()
