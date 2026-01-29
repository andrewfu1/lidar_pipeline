import numpy as np
from scipy.spatial import KDTree
from dataclasses import dataclass
from typing import Optional, List
from collections import deque


@dataclass
class ClusterResult:
    labels: np.ndarray
    num_clusters: int
    cluster_sizes: List[int]
    noise_count: int


def dbscan_cluster(
    points: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 10,
    min_cluster_size: Optional[int] = None,
    max_cluster_size: Optional[int] = None,
) -> ClusterResult:
    """
    DBSCAN clustering using KDTree
    """
    if len(points) == 0:
        return ClusterResult(
            labels=np.array([], dtype=int),
            num_clusters=0,
            cluster_sizes=[],
            noise_count=0,
        )

    xyz = points[:, :3]
    n = len(xyz)
    tree = KDTree(xyz)

    neighborhoods = tree.query_ball_point(xyz, eps)

    labels = np.full(n, -1, dtype=int)
    cluster_id = 0

    for i in range(n):
        if labels[i] != -1:
            continue
        if len(neighborhoods[i]) < min_samples:
            continue

        queue = deque(neighborhoods[i])
        labels[i] = cluster_id

        while queue:
            j = queue.popleft()
            if labels[j] == cluster_id:
                continue
            labels[j] = cluster_id
            if len(neighborhoods[j]) >= min_samples:
                queue.extend(neighborhoods[j])

        cluster_id += 1

    for cid in range(cluster_id):
        mask = labels == cid
        size = mask.sum()
        if min_cluster_size is not None and size < min_cluster_size:
            labels[mask] = -1
        elif max_cluster_size is not None and size > max_cluster_size:
            labels[mask] = -1

    unique_labels = sorted(set(labels[labels >= 0]))
    remap = {old: new for new, old in enumerate(unique_labels)}
    new_labels = np.full(n, -1, dtype=int)
    for old, new in remap.items():
        new_labels[labels == old] = new
    labels = new_labels

    num_clusters = len(unique_labels)
    cluster_sizes = [int((labels == cid).sum()) for cid in range(num_clusters)]
    noise_count = int((labels == -1).sum())

    return ClusterResult(
        labels=labels,
        num_clusters=num_clusters,
        cluster_sizes=sorted(cluster_sizes, reverse=True),
        noise_count=noise_count,
    )
