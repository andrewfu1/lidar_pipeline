import numpy as np
from dataclasses import dataclass

@dataclass
class BoundingBox:
    min_x: int
    min_y: int
    min_z: int
    max_x: int
    max_y: int
    max_z: int

@dataclass
class Detection:
    bounding_box: BoundingBox
    frame: int
    cluster_id: int
    size: int # how many points, not sure if needed

