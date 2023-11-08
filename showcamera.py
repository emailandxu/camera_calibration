import moderngl as mgl
from main import *
from camera import from_openmvg, from_colmap
from scipy.spatial.transform import Rotation


class ShowCamera(Window):
    def __init__(self, ctx: "Context" = None, wnd: "BaseWindow" = None, timer: "BaseTimer" = None, **kwargs):
        super().__init__(ctx, wnd, timer, **kwargs)
        
        self.camera.speed = 0.01
        pts, rgbs = fetchPCD("db/avenue/output/reconstruction_global/colorized.ply")
        # pts, rgbs = fetchPCD("db/avenue-cubic/sparse/0/points3D.ply")
        self.setPoints(pts, rgbs, trans=[0, 0, 0], quat=[0, 0, 0, 1])
        
        for key, mat in from_openmvg("db/avenue/output/reconstruction_global/sfm_data.json", "db/avenue/input").items():
        # for key, mat in from_colmap("db/avenue-cubic/sparse/0/images.txt", "db/avenue-cubic/images").items():
            # if not os.path.exists(os.path.join("db/avenue/output/images", key)): continue
            mat = np.array(mat)
            t, q = mat[:3, 3], Rotation.from_matrix(mat[:3, :3].transpose()).as_quat()
            t = t
            print(t, q)
            # self.setCamera(t, q, key)
            # self.setPoints(np.array(t).astype("f4"), (np.array([0, 0, 255])/255).astype("f4"), trans=[0, 0.001, 0], quat=q)

            # img = imread(key)
            # img = pad_image_to_size(img, max(*img.shape[:2]), max(*img.shape[:2]))
            # self.setPlane(img, trans=t, quat=q)
            # self.setPlane(img, t, q)

        
    

if __name__ == "__main__":
    run_window_config(ShowCamera)