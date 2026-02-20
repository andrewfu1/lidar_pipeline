# LiDAR Point Cloud Pipeline

A LiDAR processing pipeline for Velodyne point clouds with multi object tracking.

## Pipeline

**Single Frame**
1. Load raw point cloud
2. Preprocessing (voxel downsample, radial outlier removal)
3. Ground plane extraction (RANSAC)
4. Cluster obstacles (DBSCAN)

**Sequence (WIP)**
1. Process all frames in sequence
2. Extract detections from clusters
3. Associate detections across frames (Hungarian algorithm with IoU)
4. Track lifecycle management

## Run

Python 3.10+

```bash
pip install -r requirements.txt
streamlit run src/app.py
```

Two modes available:
- **Single Frame**: Analysis with preprocessing, ground segmentation, and clustering visualizations
- **Sequence**: Multi object tracking with bird's eye view and playback controls

## Data

This repo includes a small sample from the KITTI dataset for demonstration purposes.

For the full dataset, register and download from https://www.cvlibs.net/datasets/kitti/

```
data/
├── sample1.txt, sample1.png    # Individual frames for single-frame mode
├── sample2.txt, sample2.png
├── sample3.txt, sample3.png
└── 2011_09_26_drive_0001_extract/   # Sequence for tracking mode
    ├── velodyne_points/data/        # Point clouds (.txt)
    └── image_02/data/               # Camera images (.png)
```

**Citation:**
> A. Geiger, P. Lenz, R. Urtasun. "Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite." CVPR 2012.

## Future Work

- Kalman filter for motion prediction
- Oriented bounding boxes
- Object classification
