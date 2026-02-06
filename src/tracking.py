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

# This is 2d for now
def calculate_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    # no overlap edge case
    if box1.min_x >= box2.max_x or box1.min_y >= box2.max_y:
        return 0.0
    if box2.min_x >= box1.max_x or box2.min_y >= box1.max_y:
        return 0.0
    
    # zero area edge case
    box1_area = (box1.max_x - box1.min_x) * (box1.max_y - box1.min_y)
    box2_area = (box2.max_x - box2.min_x) * (box2.max_y - box2.min_y)
    if box1_area == 0 or box2_area == 0:
        return 0.0

    min_x, min_y = max(box1.min_x, box2.min_x), max(box1.min_y, box2.min_y)
    max_x, max_y = min(box1.max_x, box2.max_x), min(box1.max_y, box2.max_y)
    intersection_area = (max_x - min_x) * (max_y - min_y)

    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area

    return iou
