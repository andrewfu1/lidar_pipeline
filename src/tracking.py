import numpy as np
from dataclasses import dataclass

@dataclass
class BoundingBox:
    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float

@dataclass
class Detection:
    bounding_box: BoundingBox
    frame: int
    cluster_id: int
    size: int

def extract_detection(labels: np.ndarray, points: np.ndarray, frame: int) -> list[Detection]:
    all_detections = []

    unique_clusters = np.unique(labels)

    cluster_groups = {}
    for cluster_id in unique_clusters:
        if cluster_id == -1:
            continue
        cluster_groups[cluster_id] = points[labels == cluster_id]

    for cluster_id in cluster_groups:
        cluster_points = cluster_groups[cluster_id]

        bounding_box = BoundingBox(
            cluster_points[:, 0].min(), 
            cluster_points[:, 1].min(), 
            cluster_points[:, 2].min(),
            cluster_points[:, 0].max(),
            cluster_points[:, 1].max(),
            cluster_points[:, 2].max()
        )
        all_detections.append(Detection(bounding_box, frame, cluster_id, len(points)))

    return all_detections

