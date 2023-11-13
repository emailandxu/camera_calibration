import os
import numpy as np
from scipy.spatial.transform import Rotation
from plyfile import PlyData, PlyElement

from .camera_parser import parse

def from_colmap(path, root_path=""):
    """sparse/images.txt, trans, rotmat"""
    cameras = parse(open(path))

    return list(map(lambda c: (
        os.path.join(root_path, c.name), c.trans, Rotation.from_quat([*c.quat[1:], c.quat[0]]).as_matrix()
    ), cameras))

def from_openmvg(path, images=""):
    """reconstruction_global/sfm-data.json, center, rotmat"""
    import json
    obj = json.load(open(path))
    filemap = dict(map(
        lambda v: ( v["key"], 
                   v["value"]["ptr_wrapper"]["data"]["filename"]),
        obj["views"]
    ))
    poses = [ (os.path.join(images, filemap[o['key']]), o['value']['center'], o['value']['rotation']) 
             for o in obj['extrinsics'] ]
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
