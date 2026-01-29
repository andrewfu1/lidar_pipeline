import numpy as np
from pathlib import Path
from typing import Union


def load_kitti_txt(file_path: Union[str, Path]) -> np.ndarray:
    """
    Load a KITTI LiDAR point cloud from a .txt file
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Point cloud file not found: {file_path}")

    points = np.loadtxt(file_path, dtype=np.float32)
    return points


