from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable
import numpy as np
from src.data_loader import load_kitti_txt, discover_kitti_sequence
from src.preprocessing import voxel_downsample, radial_outlier_removal
from src.ransac import ransac_ground_plane, PlaneModel
from src.clustering import dbscan_cluster
from src.tracking import Tracker, extract_detection


@dataclass
class PipelineParams:
    """Parameters for the LiDAR processing pipeline."""
    # Preprocessing
    voxel_size: float = 0.1
    radius_scale: float = 0.03
    min_neighbors: int = 5
    # RANSAC
    ransac_iters: int = 100
    dist_thresh: float = 0.2
    normal_thresh: float = 0.9
    # Clustering
    eps: float = 0.5
    min_samples: int = 10
    min_cluster: int = 0
    max_cluster: int = 0


@dataclass
class FrameResult:
    """Result of processing a single frame."""
    # File info
    points_path: str
    image_path: Optional[str]

    # Raw/preprocessing
    raw_count: int
    downsampled_points: np.ndarray
    downsampled_count: int
    filtered_points: np.ndarray
    filtered_count: int
    outliers_removed: int

    # Ground segmentation
    plane_model: PlaneModel
    ground_points: np.ndarray
    obstacle_points: np.ndarray

    # Clustering
    cluster_labels: np.ndarray
    num_clusters: int
    cluster_sizes: list
    noise_count: int

    # Tracking
    tracks: list = field(default_factory=list)


def run_frame_pipeline(
    points_path: str,
    params: PipelineParams,
    image_path: Optional[str] = None,
) -> FrameResult:
    """
    Run the full pipeline on a single frame.
    """
    # Load
    raw_points = load_kitti_txt(points_path)

    # Downsample
    downsampled = voxel_downsample(raw_points, voxel_size=params.voxel_size)

    # Outlier removal
    filtered = radial_outlier_removal(
        downsampled,
        radius_scale=params.radius_scale,
        min_neighbors=params.min_neighbors,
    )

    # Ground segmentation
    plane, ground_mask = ransac_ground_plane(
        filtered,
        num_iterations=params.ransac_iters,
        distance_threshold=params.dist_thresh,
        normal_threshold=params.normal_thresh,
    )
    ground_points = filtered[ground_mask]
    obstacle_points = filtered[~ground_mask]

    # Clustering
    cluster_result = dbscan_cluster(
        obstacle_points,
        eps=params.eps,
        min_samples=params.min_samples,
        min_cluster_size=params.min_cluster if params.min_cluster > 0 else None,
        max_cluster_size=params.max_cluster if params.max_cluster > 0 else None,
    )

    return FrameResult(
        points_path=points_path,
        image_path=image_path,
        raw_count=len(raw_points),
        downsampled_points=downsampled,
        downsampled_count=len(downsampled),
        filtered_points=filtered,
        filtered_count=len(filtered),
        outliers_removed=len(downsampled) - len(filtered),
        plane_model=plane,
        ground_points=ground_points,
        obstacle_points=obstacle_points,
        cluster_labels=cluster_result.labels,
        num_clusters=cluster_result.num_clusters,
        cluster_sizes=cluster_result.cluster_sizes,
        noise_count=cluster_result.noise_count,
        tracks=[],
    )


def run_sequence_pipeline(
    seq_dir: str,
    params: PipelineParams,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> list[FrameResult]:
    """
    Process all frames in a KITTI sequence.
    """
    frames = discover_kitti_sequence(seq_dir)
    if not frames:
        return []

    results = []
    tracker = Tracker()

    for i, frame in enumerate(frames):
        if progress_callback:
            progress_callback(i, len(frames))

        result = run_frame_pipeline(
            points_path=str(frame["points"]),
            params=params,
            image_path=str(frame["image"]) if frame["image"] else None,
        )

        detections = extract_detection(
            labels=result.cluster_labels,
            points=result.obstacle_points,
            frame=i,
        )
        confirmed_tracks = tracker.update(detections)
        result.tracks = list(confirmed_tracks)

        results.append(result)

    if progress_callback:
        progress_callback(len(frames), len(frames))

    return results
