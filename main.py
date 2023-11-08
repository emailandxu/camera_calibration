#!python
from functools import lru_cache
import os
import imgui
import numpy as np

from imageio import imread

import moderngl as mgl
from moderngl_window import geometry, resources, run_window_config
from moderngl_window.opengl.vao import VAO

from graphics.base import WindowBase
from graphics.utils.mathutil import spherical, posemat, lookAt, projection, quat2mat, mat2quat, rotate_x, rotate_y, rotate_z

from util import pad_image_to_size
from plyfile import PlyData, PlyElement

random_tex = lambda : (np.random.rand(255, 255, 3) * 255).astype("u1")

def fetchPCD(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    return positions, colors

class FPSCamera():
    def __init__(self, speed=0.5) -> None:
        self.eye = np.array([0., 1., 1.])
        self.theta = 0.
        self.phi = np.pi * (1/5)
        self.speed = speed
        self.frame_t = 1.
        self.dragable = True

    @property
    def oriental(self):
        # return np.array(spherical(self.theta, np.pi/2, 1.))
        return self.look_target
    
    @property
    def look_target(self):
        return np.array(spherical(self.theta, self.phi, 1.))
    
    @property
    @lru_cache(maxsize=-1)
    def proj(self):
        return projection(fov=45, near=0.001)

    @property
    def view(self):
        return lookAt(eye=self.eye, at=self.eye-self.look_target, up=np.array([0, 1, 0]))

    def key_event(self, key, action, modifiers):
        if action == "ACTION_PRESS":
            if key == 119: # W
                self.eye -= self.frame_t * self.speed * self.oriental
            elif key==97: # A
                self.eye += self.frame_t * self.speed * np.cross(self.oriental, np.array([0.,1.,0.]))
            elif key==115: # S
                self.eye += self.frame_t * self.speed * self.oriental
            elif key==100: # D
                self.eye -= self.frame_t * self.speed * np.cross(self.oriental, np.array([0.,1.,0.]))
            elif key==106: # J
                self.theta+= self.frame_t
            elif key==108: # L
                self.theta-= self.frame_t
            elif key==105: # J
                self.phi+= self.frame_t
            elif key==107: # K
                self.phi-= self.frame_t
            elif key==99: # C
                self.eye[1]-= self.frame_t * self.speed
            elif key==32: # Space
                self.eye[1]+= self.frame_t * self.speed
            else:
                print(key)
    
    def mouse_scroll_event(self, x_offset, y_offset):
        # print(x_offset, y_offset)
        self.eye += self.oriental * -y_offset * self.speed

    def mouse_drag_event(self, x, y, dx, dy):
        if self.dragable:
            # print(x, y, dx, dy)
            self.theta += 0.5 * -self.frame_t * dx
            self.phi += 0.5 * -self.frame_t * dy


    def debug_gui(self):
        _, self.dragable = imgui.checkbox("camera_drag", self.dragable)
        imgui.text(f"{self.eye.astype('f2')}")
        imgui.text(f"{np.rad2deg(self.theta):.4f}, {np.rad2deg(self.phi):.4f}")

class XObjBase():
    def __init__(self, name="undefined") -> None:
        self.scale = np.ones(3)
        self.trans = np.zeros(3) # x, y, z
        self.quat = np.array([0., 0., 0., 1.]) # x, y, z, w
        self.scale_offset = np.ones(3)

        self.name = name
        self.visible = True

    @property
    def posemat(self):
        return posemat(self.trans, self.quat, self.scale)

class XObj(XObjBase):
    def __init__(self, name="undefined") -> None:
        super().__init__(name)

    def bind_vao(self, vao):
        self.vao = vao

    def bind_prog(self, prog):
        self.prog = prog

    def bind_texture(self, texture):
        self.texture = texture

    def render(self, camera, vao=None, prog=None):
        if not self.visible:
            return

        assert hasattr(camera, "view")
        assert hasattr(camera, "proj")
        
        assert vao is not None or self.vao is not None
        assert prog is not None or self.prog is not None

        if vao is not None:
            self.bind_vao(vao)

        if prog is not None:
            self.bind_prog(prog)

        if hasattr(self, "texture"):
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
        self.camera = FPSCamera()
        self.default_prog = self.load_program("default.glsl")
        self.pcd_prog = self.load_program("pcd.glsl")

    def registerCamera(self, vc, trans:np.ndarray, quat:np.ndarray, length=1.):


        vertices = np.array([0., 0., 0., -length, 0., 0.,
                                0., 0., 0., 0., length, 0.,
                                0., 0., 0., 0., 0., -length], dtype="f4").reshape(-1, 3)
        colors = np.array([ 255, 0, 255, 255, 0, 255,
                            0, 255, 0, 0, 255, 0,
                            0, 0, 255, 0, 0, 255], dtype="u1").reshape(-1, 3)
        vertices = (quat2mat(quat) @ vertices.T).T + trans
        vc.append((vertices, colors))

    def registerAxis(self, vc, trans:np.ndarray, quat:np.ndarray, length=1.):
        vertices = np.array([0., 0., 0., length, 0., 0.,
                                0., 0., 0., 0., length, 0.,
                                0., 0., 0., 0., 0., length], dtype="f4").reshape(-1, 3)
        colors = np.array([ 255, 0, 0, 255, 0, 0,
                            0, 255, 0, 0, 255, 0,
                            0, 0, 255, 0, 0, 255], dtype="u1").reshape(-1, 3)
        vertices = (quat2mat(quat) @ vertices.T).T + trans
        vc.append((vertices, colors))

    def setAxis(self, vc, name=None):
        xobj = XObj("axis" + (f"_{name}" if name else "") )
        vao = VAO(mode=mgl.LINES)

        vertices = np.concatenate([v for v,_ in vc], axis=0)
        colors = np.concatenate([c for _, c in vc], axis=0)

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
        if action=="ACTION_PRESS":
            self.xtasks[key] = lambda : self.camera.key_event(key, action, modifiers)
        else:
            self.xtasks.pop(key)

    def mouse_scroll_event(self, x_offset, y_offset):
        super().mouse_scroll_event(x_offset, y_offset)
        self.camera.mouse_scroll_event(x_offset, y_offset)

    def mouse_drag_event(self, x, y, dx, dy):
        super().mouse_drag_event(x, y, dx, dy)
        self.camera.mouse_drag_event(x, y, dx, dy)
    
    def xrender(self, t, frame_t):
        imgui.text(f"{1/frame_t:.4f}")
        self.camera.frame_t = frame_t
        self.camera.debug_gui()

        for idx, xobj in enumerate(self.xobjs):
            # imgui.text(xobj.name)
            _, xobj.visible = imgui.checkbox(str(idx) + ":" + xobj.name, xobj.visible)
            imgui.same_line()

        assert len(self.xobjs) > 0

        for xtask in self.xtasks.values():
            xtask()

        for xobj in self.xobjs:
            # xobj.quat = mat2quat(rotate_x(t) @ rotate_y(t))
            xobj.render(self.camera)


