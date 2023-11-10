import os
import numpy as np
from scipy.spatial.transform import Rotation
from plyfile import PlyData, PlyElement

from .camera_parser import parse

def pose2mat(t, R):
    return np.array([
        [*R[0], t[0]],
        [*R[1], t[1]],
        [*R[2], t[2]],
        [ 0, 0, 0, 1]
    ], dtype="f4")

def from_colmap(path, root_path=""):

    def colmap_viewmat(t, R):
        return np.array([
            [*R[0], t[0]],
            [*R[1], t[1]],
            [*R[2], t[2]],
            [ 0, 0, 0, 1]
        ], dtype="f4")

    """sparse/images.txt"""
    cameras = parse(open(path))
    poses = dict(map(lambda c: (os.path.join(root_path, c.name), colmap_viewmat(
        c.trans,
        Rotation.from_quat([*c.quat[1:], c.quat[0]]).as_matrix()
    )), cameras))
    return poses

def from_openmvg(path, images=""):
    """reconstruction_global/sfm-data.json"""
    import json
    obj = json.load(open(path))
    filemap = dict(map(
        lambda v: ( v["key"], 
                   v["value"]["ptr_wrapper"]["data"]["filename"]),
        obj["views"]
    ))
    poses = dict(map(lambda o: (os.path.join(images, filemap[o['key']]), pose2mat(
        o['value']['center'], o['value']['rotation']
    )), obj['extrinsics']))
    return poses

def fetchPCD(path):
    """return positions and color"""
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    return positions, colors


if __name__ == "__main__":
    # poses = from_colmap("/data1/xushuli/git-repo/gaussian-splatting/db/fan/sparse/images.txt")
    poses = from_openmvg("/data1/xushuli/git-repo/gaussian-splatting/db/ambu/output/reconstruction_global/sfm-data.json")
    print(poses.keys())
    poses = np.array(list(poses.values())).reshape(-1, 4, 4)
    print(poses.shape)
