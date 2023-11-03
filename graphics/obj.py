import moderngl
import numpy as np
from moderngl_window.opengl.vao import VAO
from .utils.mathutil import *


class XObjBase():
    def __init__(self) -> None:
        self.scale = np.ones(3)
        self.trans = np.zeros(3) # x, y, z
        self.quat = np.array([0., 0., 0., 1.]) # x, y, z, w

    @property
    def posemat(self):
        return posemat(self.trans, self.quat, self.scale)


class XObj(XObjBase):
    def __init__(self, vao:"VAO", ctx:"moderngl.Context", image:"np.ndarray"=None) -> None:
        super().__init__()
        
        self.vao = vao
        self.window = window
        self.prog = window.load_program("default.glsl")
        
        if image is None:
            self.image = (np.ones((1, 1, 3)) * 255 ).astype("u1")
        else:
            assert image.dtype == np.uint8 
            assert len(image.shape) == 3
            self.image = image
    
        size, components = self.image.shape[:2], self.image.shape[-1]
        self.texture : mgl.Texture = window.ctx.texture(size, components)

    def render(self):
        self.texture.write(self.image)
        self.texture.use(0)
        self.prog['texture0'].value = 0

        m, v, p = self.posemat, self.window.view, self.window.proj

        # note that transpose is essential, don't know why.
        mvp = (p @ v @ m).transpose().astype("f4").copy()
        
        self.prog["mvp"].write(mvp)
        self.vao.render(self.prog)
