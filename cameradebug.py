from main import *
from scipy.spatial.transform import Rotation
import re

def mat2TQ(mat):
    mat = np.array(mat)
    trans = mat[:3, 3] * 3
    quat =  Rotation.from_matrix(mat[:3, :3]).as_quat()
    return trans, quat

def Q2mat(q):
    return Rotation.from_quat(q).as_matrix()

def mat2Q(mat):
    return Rotation.from_matrix(mat).as_quat()

class CameraCalibrationWindow(Window):
    def __init__(self, ctx: "mgl.Context" = None, wnd: "BaseWindow" = None, timer: "BaseTimer" = None, **kwargs):
        super().__init__(ctx, wnd, timer, **kwargs)
        from camera import from_openmvg, from_colmap
        from scipy.spatial.transform import Rotation
        openmvg_path = "db/avenue/output/sfm_data_perspective.json"


        openmvg_dict = from_openmvg(openmvg_path)
        colmap_dict = from_colmap("db/avenue-cubic/sparse/0/images.txt", "")


        mat_src_target_dict = {}
        for key in openmvg_dict.keys() & colmap_dict.keys():
            head = re.sub("_\d+\.png", "", key)
            mat_src, mat_target = np.array(openmvg_dict[key]), np.array(colmap_dict[key])
            mat_src_target_dict[head] = (mat_src, mat_target)

        for key in sorted(openmvg_dict.keys()):
            if not os.path.exists(os.path.join("db/avenue/output/images", key)):
                continue

            head = re.sub("_\d+\.png", "", key)
            if head not in mat_src_target_dict:
                continue
            print(key, head)
            mat = np.array(openmvg_dict[key])
            mat_src, mat_target = mat_src_target_dict[head]
            t = mat_target[:3, 3]    
            # _, q = mat2TQ( mat @ np.linalg.inv(mat_src) @ mat_target)
            q = mat2Q(mat[:3, :3].transpose())
            print(q[[3, 0, 1, 2]])
            self.setAxis(t, q)



if __name__ == "__main__":
    run_window_config(CameraCalibrationWindow, args=["--window", "pyglet"])