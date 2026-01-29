import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class PlaneModel:
    """
    Represents a 3D plane: normal * point + d = 0
    """
    # Unit vector with distance parameter to represent plane
    normal: np.ndarray
    d: float

    def distance_to_points(self, points: np.ndarray) -> np.ndarray:
        return np.abs(np.dot(points[:, :3], self.normal) + self.d)

    @property
    def equation_string(self) -> str:
        return f"{self.normal[0]:.4f}x + {self.normal[1]:.4f}y + {self.normal[2]:.4f}z + {self.d:.4f} = 0"


def fit_plane_from_points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> PlaneModel:
    """
    Fit a plane through three 3D points.
    """
    v1 = p2 - p1
    v2 = p3 - p1

    normal = np.cross(v1, v2)

    norm = np.linalg.norm(normal)
    if norm < 1e-10:
        raise ValueError("Points are collinear")

    normal = normal / norm

    d = -np.dot(normal, p1)

    return PlaneModel(normal=normal, d=d)


def ransac_ground_plane(
    points: np.ndarray,
    num_iterations: int = 100,
    distance_threshold: float = 0.2,
    normal_threshold: float = 0.9,
) -> Tuple[PlaneModel, np.ndarray]:
    """
    Detect ground plane using RANSAC.
    We are not using height filtering right now.
    """
    xyz = points[:, :3]
    n_points = len(xyz)

    if n_points < 3:
        raise ValueError(f"Need at least 3 points")

    best_plane = None
    best_inlier_count = 0
    best_inlier_mask = np.zeros(n_points, dtype=bool)

    for _ in range(num_iterations):
        sample_indices = np.random.choice(n_points, 3, replace=False)
        p1, p2, p3 = xyz[sample_indices]

        try:
            plane = fit_plane_from_points(p1, p2, p3)
        except ValueError:
            continue

        if abs(plane.normal[2]) < normal_threshold:
            continue

        if plane.normal[2] < 0:
            plane.normal = -plane.normal
            plane.d = -plane.d

        distances = plane.distance_to_points(xyz)
        inlier_mask = distances < distance_threshold
        inlier_count = np.sum(inlier_mask)

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_plane = plane
            best_inlier_mask = inlier_mask

    if best_plane is None:
        raise RuntimeError(f"RANSAC failed after {num_iterations} iterations")

    return best_plane, best_inlier_mask
