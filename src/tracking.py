import numpy as np
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment
from enum import Enum

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

class TrackStatus(Enum):
    TENTATIVE = 1
    CONFIRMED = 2
    LOST = 3

@dataclass
class Track:
    track_id: int
    bounding_box: BoundingBox
    n_init: int = 3
    max_age: int = 5
    age: int = 1
    hits: int = 1
    misses: int = 0
    status: TrackStatus = TrackStatus.TENTATIVE

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
        all_detections.append(Detection(bounding_box, frame, cluster_id, len(cluster_points)))

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

def object_assignment(detections: list[Detection], tracks: list[Track]):
    n = max(len(detections), len(tracks))
    cost_matrix = np.zeros((n, n))

    for i in range(len(detections)):
        for j in range(len(tracks)):
            detection = detections[i]
            track = tracks[j]
            iou = calculate_iou(detection.bounding_box, track.bounding_box)
            cost_matrix[i][j] = iou

    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)

    output_pair = []

    for i in range(len(row_ind)):
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
    for i in range(len(detections)):
        if i not in matched_detections:
            unmatched_detections.append(i)

    unmatched_tracks = []
    for i in range(len(tracks)):
        if i not in matched_tracks:
            unmatched_tracks.append(i)
    
    return (output_pair, unmatched_detections, unmatched_tracks)
            

class Tracker:
    def __init__(self, n_init: int = 3, max_age: int = 5, iou_threshold: float = 0.25):
        self.next_id = 0
        self.all_tracks: list[Track] = []
        self.n_init = n_init
        self.max_age = max_age
        self.iou_threshold = iou_threshold

    def update(self, detections: list[Detection]) -> list[Track]:
        # match detections to existing tracks
        matches, unmatched_dets, unmatched_tracks = object_assignment(detections, self.all_tracks)

        # update matched tracks
        for det_idx, track_idx in matches:
            track = self.all_tracks[track_idx]
            detection = detections[det_idx]
            track.bounding_box = detection.bounding_box
            track.hits += 1
            track.misses = 0
            track.age += 1
            if track.status == TrackStatus.TENTATIVE and track.hits >= track.n_init:
                track.status = TrackStatus.CONFIRMED

        # mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            track = self.all_tracks[track_idx]
            track.misses += 1
            track.age += 1
            if track.misses > track.max_age:
                track.status = TrackStatus.LOST

        # create new  tracks
        for det_idx in unmatched_dets:
            new_track = Track(
                track_id=self.next_id,
                bounding_box=detections[det_idx].bounding_box,
                n_init=self.n_init,
                max_age=self.max_age
            )
            self.next_id += 1
            self.all_tracks.append(new_track)

        # prune lost tracks
        active_tracks = []
        for track in self.all_tracks:
            if track.status != TrackStatus.LOST:
                active_tracks.append(track)
        self.all_tracks = active_tracks

        confirmed_tracks = []
        for track in self.all_tracks:
            if track.status == TrackStatus.CONFIRMED:
                confirmed_tracks.append(track)

        return confirmed_tracks