#%%
import trimesh
import numpy as np

#%%
def from_colmap(path):
    """sparse/images.txt"""
    from exp.camera_parser import parse
    cameras = parse(open(path))
    points = [c.trans for c in cameras]
    return points
#%%
def from_openmvg(path):
    """reconstruction_global/sfm-data.json"""
    import json
    obj = json.load(open(path))
    points = list(map(lambda o: o['value']['center'], obj['extrinsics']))
    return points
#%%
# points = from_colmap("/data1/xushuli/git-repo/gaussian-splatting/db/fan/sparse/images.txt")
points = from_openmvg("/data1/xushuli/git-repo/gaussian-splatting/db/ambu/output/reconstruction_global/sfm-data.json")
points = np.array(points).reshape(-1,3)
#%%
