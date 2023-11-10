from functools import lru_cache
import numpy as np

def spherical(theta, phi, radius):
    # https://community.khronos.org/t/moving-the-camera-using-spherical-coordinates/49549
    # theta = np.radians(theta)
    # phi = np.radians(phi)
    y = radius * np.sin(phi)
    x = radius * np.cos(phi) * np.cos(theta)
    z = radius * np.cos(phi) * np.sin(theta)
    return x, y, z

def cartesian(x,y,z, radius=1):
    phi = np.arcsin(y/radius)
    theta = np.arccos(x / (radius * np.cos(phi)))
    return theta, phi

def lookAt(eye, at, up):
    a = eye - at
    b = up
    w = a / np.linalg.norm(a)
    u = np.cross(b, w)
    u = u / np.linalg.norm(u)
    v = np.cross(w, u)
    translate = np.array([[1, 0, 0, -eye[0]], 
                          [0, 1, 0, -eye[1]], 
                          [0, 0, 1, -eye[2]], 
                          [0, 0, 0, 1]]).astype(np.float32)
    rotate =  np.array([[u[0], u[1], u[2], 0], 
                        [v[0], v[1], v[2], 0], 
                        [w[0], w[1], w[2], 0], 
                        [0, 0, 0, 1]]).astype(np.float32)
    # return np.matmul(rotate, translate)
    return rotate @ translate

def projection(fov=90.0, aspect_ratio=1.0, near=0.1, far=1000.0):
    # Convert the fov from degrees to radians
    fov_rad = np.deg2rad(fov)
    
    # Calculate the scale using the fov
    scale = 1.0 / np.tan(fov_rad / 2.0)
    
    # Create the projection matrix
    return np.array([[scale / aspect_ratio, 0, 0, 0],
                     [0, scale, 0, 0],
                     [0, 0, -(far+near) / (far - near), -2 * far * near / (far - near)],
                     [0, 0, -1, 0]]).astype(np.float32)

                    
class FPSCamera():
    def __init__(self, speed=0.5) -> None:
        self.eye = np.array([0., 1., 1.])
        self.theta = 0.
        self.phi = np.pi * (1/5)
        self.speed = speed
        self.frame_t = 1.
        self.dragable = False

    @property
    def oriental(self):
        return np.array(spherical(self.theta, self.phi, 1.))
    
    @property
    @lru_cache(maxsize=-1)
    def proj(self):
        return projection(fov=45, near=0.001)

    @property
    def view(self):
        return lookAt(eye=self.eye, at=self.eye+self.oriental, up=np.array([0, 1, 0]))
    
    @view.setter
    def view(self, newview:np.ndarray):
        x, y, z = -newview[:3,:3].transpose()[2, :3] # z_axis
        self.theta, self.phi = cartesian(x, y, z)
        self.eye = -newview[:3, 3]


    def key_event(self, key, action, modifiers):
        if key == 119: # W
            self.eye += self.frame_t * self.speed * self.oriental
        elif key==97: # A
            self.eye -= self.frame_t * self.speed * np.cross(self.oriental, np.array([0.,1.,0.]))
        elif key==115: # S
            self.eye -= self.frame_t * self.speed * self.oriental
        elif key==100: # D
            self.eye += self.frame_t * self.speed * np.cross(self.oriental, np.array([0.,1.,0.]))
        elif key==106: # J
            self.theta -= self.frame_t
        elif key==108: # L
            self.theta += self.frame_t
        elif key==105: # I
            self.phi += self.frame_t
        elif key==107: # K
            self.phi -= self.frame_t
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

    def debug_gui(self, t, frame_t):
        import imgui
        from ..widgets import float_widget
        self.frame_t = frame_t
        if not hasattr(self, "wspeed"):
            self.wspeed = float_widget("speed", 0.01, 1, 0.1)
        self.speed = self.wspeed()
        _, self.dragable = imgui.checkbox("camera_drag", self.dragable)
        imgui.text(f"{self.eye.astype('f2')}")
        imgui.text(f"{np.rad2deg(self.theta):.4f}, {np.rad2deg(self.phi):.4f}")
        imgui.text(f"{self.eye - self.oriental}")
