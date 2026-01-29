import numpy as np
from scipy.spatial import KDTree

def voxel_downsample(points: np.ndarray, voxel_size: float = 0.1) -> np.ndarray:
    """
    Downsample point cloud using voxel grid filtering.
    """
    if len(points) == 0:
        return np.zeros((0, 3))

    xyz = points[:, :3].astype(np.float64)

    # Compute voxel indices for each point
    # Then shift indices to handle negative values
    # Then create unique hash for each voxel
    # Find unique voxels and compute centroids
    # Compute centroid for each voxel
    voxel_indices = np.floor(xyz / voxel_size).astype(np.int64)

    min_indices = voxel_indices.min(axis=0)
    shifted_indices = voxel_indices - min_indices

    max_dim = shifted_indices.max(axis=0) + 1
    voxel_hash = (shifted_indices[:, 0] * (max_dim[1] * max_dim[2]) + shifted_indices[:, 1] * max_dim[2] + shifted_indices[:, 2])

    unique_hashes, inverse_indices = np.unique(voxel_hash, return_inverse=True)
    num_voxels = len(unique_hashes)

    counts = np.bincount(inverse_indices)
    centroids = np.zeros((num_voxels, 3))

    for dim in range(3):
        centroids[:, dim] = np.bincount(inverse_indices, weights=xyz[:, dim]) / counts

    return centroids


def radial_outlier_removal(
    points: np.ndarray,
    radius_scale: float = 0.03,
    min_neighbors: int = 5
) -> np.ndarray:
    """
    Remove outlier points using radius from sensor. Search radius should scale with distance.
    """
    if len(points) == 0:
        return points.copy()
    
    xyz = points[:, :3]

    radial_distances = np.linalg.norm(xyz, axis=1)
    radial_distances = np.maximum(radial_distances, 0.1)

    tree = KDTree(xyz)

    keep_mask = np.zeros(len(points), dtype=bool)

    for i in range(len(points)):
        search_radius = radius_scale * radial_distances[i]

        neighbors = tree.query_ball_point(xyz[i], search_radius)

        if len(neighbors) - 1 >= min_neighbors:
            keep_mask[i] = True

    return points[keep_mask]
