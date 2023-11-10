
from typing import List
import imgui
import numpy as np


import moderngl as mgl
from moderngl_window import geometry, resources, run_window_config
from moderngl_window.opengl.vao import VAO

from .xobj import XObj
from .base import WindowBase
from .camera import FPSCamera

from .utils.mathutil import *
from .utils.meshutil import makeCoord, applyMat
from .widgets import *

class Window(WindowBase):
    def __init__(self, ctx: "mgl.Context" = None, wnd: "BaseWindow" = None, timer: "BaseTimer" = None, **kwargs):
        super().__init__(ctx, wnd, timer, **kwargs)
        self.xobjs = []
        self.xtasks = {}
        self.camera = FPSCamera()
        self.default_prog = self.load_program("default.glsl")
        self.pcd_prog = self.load_program("pcd.glsl")

    def setAxis(self, vertices:List[np.ndarray], colors:np.ndarray=None, name=None):
        """assume vertices and colors in shape (n, 3, 6) """

        vertices = np.concatenate(vertices, axis=0).reshape(-1, 3, 6)
        assert len(vertices.shape) == 3 and vertices.shape[1:] == (3, 6)

        if colors is None:
            n = vertices.shape[0]
            colors = np.array([[[1, 0, 0] * 2, [0, 1, 0] * 2, [0, 0, 1] * 2]], dtype="f4").repeat(n, axis=0) # n x 3 x 6

        xobj = XObj("axis" + (f"_{name}" if name else "") )
        vao = VAO(mode=mgl.LINES)

        vao.buffer(np.array(vertices, dtype="f4"), '3f', 'in_position')
        vao.buffer(np.array(colors,  dtype="f4"), '3f', 'in_rgb')

        xobj.bind_vao(vao)
        xobj.bind_prog(self.pcd_prog)       
        self.xobjs.append(xobj)
        return xobj
    
    def setPlane(self, tex, trans=None, quat=None, scale=None, name=None):
        xobj = XObj("plane" + (f"_{name}" if name else ""))

        vao = geometry.quad_fs()
        texture = self.ctx.texture(tex.shape[:2], tex.shape[2], data=np.ascontiguousarray(tex[::-1]))

        xobj.bind_vao(vao)
        xobj.bind_prog(self.default_prog)
        xobj.bind_texture(texture)

        xobj.scale = scale if scale is not None else np.array([1, 1, 1])
        xobj.trans = trans if trans is not None else np.array([0, 0, -3])
        xobj.quat = quat if quat is not None else np.array([0, 0, 0, 1])

        self.xobjs.append(xobj)
        return xobj
    
    def setPoints(self, points, rgbs, trans=None, quat=None, scale=None, name=None):
        xobj = XObj("points" + (f"_{name}" if name else ""))
        vao = VAO(mode=mgl.POINTS)
        vao.buffer(np.array(points, dtype="f4"), "3f", "in_position")
        vao.buffer(np.array(rgbs, dtype="f4"), "3f", "in_rgb")

        xobj.bind_vao(vao)
        xobj.bind_prog(self.pcd_prog)

        xobj.scale = scale if scale is not None else np.array([1, 1, 1])
        xobj.trans = trans if trans is not None else np.array([0, 0, -3])
        xobj.quat = quat if quat is not None else np.array([0, 0, 0, 1])

        self.xobjs.append(xobj)
        return xobj

    def setGround(self, width=10, height=10, grid=100):
        x = np.linspace(-width//2, +width//2, grid)
        y = np.zeros_like(x)
        z_forward = np.ones_like(x) * height // 2
        z_backward = np.ones_like(x) * -height // 2
        forward = np.stack([x, y, z_forward], axis=-1)
        backward = np.stack([x, y, z_backward], axis=-1)

        z = np.linspace(-width//2, +width//2, grid)
        y = np.zeros_like(z)
        x_left = np.ones_like(z) * -width // 2
        x_right = np.ones_like(z) * width // 2
        left = np.stack([x_left, y, z], axis=-1)
        right = np.stack([x_right, y, z], axis=-1)

        vertices = np.zeros((grid*4, 3))
        vertices[0::4] = forward
        vertices[1::4] = backward
        vertices[2::4] = left
        vertices[3::4] = right

        xobj = XObj("ground")
        vao = VAO(mode=mgl.LINES)
        vao.buffer(np.array(vertices, dtype="f4"), '3f', 'in_position')
        vao.buffer(np.array(np.ones_like(vertices),  dtype="f4"), '3f', 'in_rgb')

        xobj.bind_vao(vao)
        xobj.bind_prog(self.pcd_prog)       
        self.xobjs.append(xobj)
        return xobj


    def key_event(self, key, action, modifiers):
        super().key_event(key, action, modifiers)
        if action=="ACTION_PRESS" or action == 768:
            self.xtasks[key] = lambda : self.camera.key_event(key, action, modifiers)
        elif key in self.xtasks:
            self.xtasks.pop(key)
        else:
            print(key, action)
            
    def mouse_scroll_event(self, x_offset, y_offset):
        super().mouse_scroll_event(x_offset, y_offset)
        self.camera.mouse_scroll_event(x_offset, y_offset)

    def mouse_drag_event(self, x, y, dx, dy):
        super().mouse_drag_event(x, y, dx, dy)
        self.camera.mouse_drag_event(x, y, dx, dy)
    
    def xrender(self, t, frame_t):
        imgui.text(f"{1/frame_t:.4f}")
        self.camera.debug_gui(t, frame_t)

        for idx, xobj in enumerate(self.xobjs):
            # imgui.text(xobj.name)
            _, xobj.visible = imgui.checkbox(str(idx) + ":" + xobj.name, xobj.visible)
            imgui.same_line()

        assert len(self.xobjs) > 0

        for xtask in self.xtasks.values():
            xtask()

        for xobj in self.xobjs:
            # xobj.quat = mat2quat(rotate_x(t) @ rotate_y(t))
            xobj.render(self.camera.view, self.camera.proj)


