import numpy as np
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment

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

# this is 2d for now
def calculate_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    if box1.min_x >= box2.max_x or box1.min_y >= box2.max_y:
        return 0.0
    if box2.min_x >= box1.max_x or box2.min_y >= box1.max_y:
        return 0.0
    
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

def object_assignment(detections_current: list[Detection], detections_prev: list[Detection]):
    n = max(len(detections_current), len(detections_prev))
    cost_matrix = np.zeros((n, n))

    for i in range(len(detections_current)):
        for j in range(len(detections_prev)):
            curr_detection = detections_current[i]
            prev_detection = detections_prev[j]
            iou = calculate_iou(curr_detection.bounding_box, prev_detection.bounding_box)
            cost_matrix[i][j] = iou

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    output_pair = []

    for i in range(row_ind):
        detection_idx = row_ind[i]
        track_idx = col_ind[i]
        if cost_matrix[detection_idx][track_idx] > 0.25:
            output_pair.append((detection_idx,track_idx))

    matched_detections = set()
    matched_tracks = set()
    for pair in output_pair:
        matched_detections.add(pair[0])
        matched_tracks.add(pair[1])

    unmatched_detections = []
    for i in range(len(detections_current)):
        if i not in matched_detections:
            unmatched_detections.append(i)

    unmatched_tracks = []
    for i in range(len(detections_prev)):
        if i not in matched_tracks:
            unmatched_tracks.append(i)
    

    return (matched_tracks, unmatched_detections, unmatched_tracks)
            
        