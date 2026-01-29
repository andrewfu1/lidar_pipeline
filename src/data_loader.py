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


def load_kitti_bin(file_path: Union[str, Path]) -> np.ndarray:
    """
    Load a KITTI LiDAR point cloud from a .bin
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Point cloud file not found: {file_path}")

    if file_path.suffix != '.bin':
        raise ValueError(f"Expected .bin file, got: {file_path.suffix}")

    points = np.fromfile(file_path, dtype=np.float32)

    if len(points) % 4 != 0:
        raise ValueError(
            f"Invalid file size: {len(points)} floats not divisible by 4. "
            f"Expected format: [x, y, z, reflectance] per point."
        )

    points = points.reshape(-1, 4)

    return points
