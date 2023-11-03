from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Point2D():
    x:float
    y:float
    id:int

    def __post_init__(self):
        self.x = float(self.x)
        self.y = float(self.y)
        self.id = int(self.id)

    @property
    def coord(self):
        return (self.x, self.y)
    
@dataclass
class Pose():
    quat:List[float]
    trans:List[float]


@dataclass
class ColmapCamera():
    image_id:int
    pose:List["Pose"]
    camera_id:int
    name:str
    points:List["Point2D"]

    def __repr__(self) -> str:
        fields = self.__dataclass_fields__
        addself = lambda f:"self."+f
        fields_repr = ", ".join([
            f"{f}=" + str(getattr(self, f)) if f != "points" else f"points={self.points[0]}..."
            for fid, f in enumerate(fields)
        ])
        return f"{self.__class__.__qualname__}({fields_repr})"

    @property
    def quat(self):
        return self.pose.quat
    
    @property
    def trans(self):
        return self.pose.trans
    
def parse(lines):
    def f():
        no_comment = filter(lambda line:not line.startswith("#"), lines)
        splited = list(map(lambda line: line.strip().split(" "), no_comment))
        for idx in range(len(splited) // 2):
            camera_arr = splited[idx * 2]
            point_arr = splited[idx*2 + 1]
            yield ColmapCamera(
                image_id = int(camera_arr[0]),
                pose = Pose(quat = list(map(float, camera_arr[1:5])), trans = list(map(float, camera_arr[5:8]))),
                camera_id = int(camera_arr[8]),
                name = str(camera_arr[9]),
                points = [Point2D(*point_arr[pid*3:(pid+1)*3]) for pid in range(len(point_arr)//3)]
            )
    return list(f())

if __name__== "__main__":
    lines = open("resources/sparse/0/images.txt")
    cameras = parse(lines)
    print(cameras[0])