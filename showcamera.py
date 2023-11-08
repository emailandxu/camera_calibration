import moderngl as mgl
from main import *
from camera import from_openmvg, from_colmap
from scipy.spatial.transform import Rotation
from graphics.widgets import float_widget, bool_widget, float3_widget, float4_widget


class ShowCamera(Window):
    def __init__(self, ctx: "Context" = None, wnd: "BaseWindow" = None, timer: "BaseTimer" = None, **kwargs):
        super().__init__(ctx, wnd, timer, **kwargs)
        self.wspeed = float_widget("speed", 0.01, 1, 0.1)
        self.camera.eye = np.array([0, 0.2, 0.2])

        self.wtrans1 = float3_widget("pcd1", -1, 1, (0., 0., 0.))
        self.wtrans2 = float3_widget("pcd2", -1, 1, (0., 0., 0.))
        self.wquat = float4_widget("quat", -np.pi, np.pi, (0., 0., 0., 1.))

        self.setGround()
        vc = []
        self.registerAxis(vc, np.zeros(3), np.array([0, 0, 0, 1],dtype="f4"), length=5)
        self.origin_axis_xobj = self.setAxis(vc, name="origin")

        vc = []
        self.registerCamera(vc, self.camera.eye, mat2quat(np.linalg.inv(self.camera.view[:3, :3])), length=0.1)
        self.camera_axis_xobj = self.setAxis(vc, name="camera")

        # =================== COLMAP ========================
        scale = 0.01
        pts, rgbs = fetchPCD("db/avenue-cubic/sparse/0/points3D.ply")
        self.point_xobj2 = self.setPoints(pts, rgbs, trans=[0, 0, 0], quat=[0, 0, 0, 1], scale=np.ones(3) * scale)

        vc = []
        for key, mat in from_colmap("db/avenue-cubic/sparse/0/images.txt", "db/avenue-cubic/images").items():
            mat = np.array(mat)
            t, q = mat[:3, 3], Rotation.from_matrix(mat[:3, :3]).as_quat()
            t = t
            print("!", t, q)
            self.registerCamera(vc, t * scale, q, length=0.001)
        self.setAxis(vc, "pc1")

        # =================== OPENMVG ========================
        offset = np.array([0.5, 0.5, -0.3])
        pts, rgbs = fetchPCD("db/avenue/output/reconstruction_global/colorized.ply")
        self.point_xobj1 = self.setPoints(pts + offset, rgbs, trans=[0, 0, 0], quat=[0, 0, 0, 1])

        vc = []
        for key, mat in from_openmvg("db/avenue/output/reconstruction_global/sfm_data.json", "db/avenue/input").items():
            # if not os.path.exists(os.path.join("db/avenue/output/images", key)): continue
            mat = np.array(mat)
            t, q = mat[:3, 3], Rotation.from_matrix(mat[:3, :3]).as_quat()
            t = t + offset
            print("?", t, q)
            self.registerCamera(vc, t, q, length=0.001)

            # img = imread(key)
            # img = pad_image_to_size(img, max(*img.shape[:2]), max(*img.shape[:2]))
            # self.setPlane(img, trans=t, quat=q, scale=np.ones(3)*0.001)
        self.setAxis(vc, name="pc2")

    def xrender(self, t, frame_t):
        self.camera.speed = self.wspeed()
        super().xrender(t, frame_t)

        imgui.begin("123")
        self.point_xobj1.trans = np.array(self.wtrans1())
        self.point_xobj2.trans = np.array(self.wtrans2())
        self.origin_axis_xobj.quat = self.wquat()
        imgui.end()


if __name__ == "__main__":
    run_window_config(ShowCamera)