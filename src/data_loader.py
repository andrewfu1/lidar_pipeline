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


def discover_kitti_sequence(sequence_dir: Union[str, Path]) -> list[dict]:
    """
    Discover all frames in a KITTI sequence directory.
    Returns list of dicts with 'points' and 'image' paths.
    """
    sequence_dir = Path(sequence_dir)
    velodyne_dir = sequence_dir / "velodyne_points" / "data"
    image_dir = sequence_dir / "image_02" / "data"

    if not velodyne_dir.exists():
        return []

    frames = []
    for txt_file in sorted(velodyne_dir.glob("*.txt")):
        png_file = image_dir / f"{txt_file.stem}.png"
        frames.append({
            "points": txt_file,
            "image": png_file if png_file.exists() else None
        })
    return frames
