# LiDAR Point Cloud Pipeline

Implements the preprocessing and segmentation stages of a LiDAR processing pipeline for Velodyne point clouds from the [KITTI dataset](https://www.cvlibs.net/datasets/kitti/). Tracking is WIP

## Pipeline

1. Load raw point cloud (3 sample frames from KITTI 2011_09_26 drive sequence)
2. Preprocessing (voxel downsample and remove outliers using radius)
3. Ground plane extraction (RANSAC)
4. Cluster obstacles (DBSCAN)
5. Visualize

Tracking is WIP

## Run

Python 3.10+

```bash
pip install -r requirements.txt

streamlit run src/app.py
```
Select sample point cloud from the dropdown, adjust parameters (voxel size, ransac iterations), and click run pipeline.

## Future work

Next steps for single frame processing:
- Fitting bounding boxes around clusters
- Object classification

Then multi frame pipeline:
- Run pipeline across consecutive frames
- Associate and track objects in a drive sequence (can use Kalman filters)
- Correct for vehicle movement during scan
