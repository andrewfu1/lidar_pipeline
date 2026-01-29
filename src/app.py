import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import numpy as np
import plotly.graph_objects as go

from src.data_loader import load_kitti_txt, load_kitti_bin
from src.preprocessing import voxel_downsample, radial_outlier_removal
from src.ransac import ransac_ground_plane
from src.clustering import dbscan_cluster

CLUSTER_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9A6324", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#ffffff",
]


def _scatter2d(points_list, names, colors, title, xlabel, ylabel, x_idx=0, y_idx=1):
    fig = go.Figure()
    for pts, name, color in zip(points_list, names, colors):
        if len(pts) > 0:
            fig.add_trace(go.Scattergl(
                x=pts[:, x_idx], y=pts[:, y_idx],
                mode="markers",
                marker=dict(size=2, color=color, opacity=0.5),
                name=name,
            ))
    fig.update_layout(
        title=title,
        xaxis=dict(title=xlabel, scaleanchor="y"),
        yaxis=dict(title=ylabel),
        height=550,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def _scatter3d_ground_obstacle(ground, obstacle):
    fig = go.Figure()
    if len(ground) > 0:
        fig.add_trace(go.Scatter3d(
            x=ground[:, 0], y=ground[:, 1], z=ground[:, 2],
            mode="markers",
            marker=dict(size=1, color="blue", opacity=0.4),
            name=f"Ground ({len(ground):,})",
        ))
    if len(obstacle) > 0:
        fig.add_trace(go.Scatter3d(
            x=obstacle[:, 0], y=obstacle[:, 1], z=obstacle[:, 2],
            mode="markers",
            marker=dict(size=1, color="red", opacity=0.6),
            name=f"Obstacles ({len(obstacle):,})",
        ))
    fig.update_layout(
        scene=dict(aspectmode="data"),
        height=600,
        margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig


def _scatter3d_clusters(obstacle_points, labels):
    fig = go.Figure()
    unique = np.unique(labels)
    for label in unique:
        mask = labels == label
        pts = obstacle_points[mask]
        if label == -1:
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers",
                marker=dict(size=1, color="gray", opacity=0.2),
                name=f"Noise ({len(pts):,})",
            ))
        else:
            color = CLUSTER_COLORS[label % len(CLUSTER_COLORS)]
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers",
                marker=dict(size=2, color=color, opacity=0.7),
                name=f"Cluster {label} ({len(pts):,})",
            ))
    fig.update_layout(
        scene=dict(aspectmode="data"),
        height=600,
        margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig


def render_preprocessing(r):
    st.image(r["image_file"], caption="Left color camera (image_02)", use_container_width=True)

    st.caption(
        f"Downsampled: **{r['downsampled_count']:,}** pts | "
        f"Filtered: **{r['filtered_count']:,}** pts | "
        f"Removed: **{r['outliers_removed']:,}** outliers"
    )

    col_left, col_right = st.columns(2)
    with col_left:
        fig = _scatter2d(
            [r["downsampled_points"]],
            ["Downsampled"],
            ["#4363d8"],
            "Before Outlier Removal",
            "X (m)", "Y (m)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        fig = _scatter2d(
            [r["filtered_points"]],
            ["Filtered"],
            ["#3cb44b"],
            "After Outlier Removal",
            "X (m)", "Y (m)",
        )
        st.plotly_chart(fig, use_container_width=True)


def render_ground(r):
    view = st.radio("View", ["Bird's Eye (XY)", "Side View (XZ)"], horizontal=True)

    ground = r["ground_points"]
    obstacle = r["obstacle_points"]

    if view == "Bird's Eye (XY)":
        fig = _scatter2d(
            [ground, obstacle],
            [f"Ground ({len(ground):,})", f"Obstacles ({len(obstacle):,})"],
            ["blue", "red"],
            "Ground vs Obstacles - Bird's Eye",
            "X (m)", "Y (m)",
        )
    else:
        fig = _scatter2d(
            [ground, obstacle],
            [f"Ground ({len(ground):,})", f"Obstacles ({len(obstacle):,})"],
            ["blue", "red"],
            "Ground vs Obstacles - Side View",
            "X (m)", "Z (m)",
            x_idx=0, y_idx=2,
        )
        plane = r["plane_model"]
        if plane is not None and abs(plane.normal[2]) > 1e-6:
            x_line = np.linspace(-50, 50, 100)
            z_line = (-plane.normal[0] * x_line - plane.d) / plane.normal[2]
            fig.add_trace(go.Scattergl(
                x=x_line, y=z_line,
                mode="lines",
                line=dict(color="green", width=3),
                name="Ground Plane",
            ))

    st.plotly_chart(fig, use_container_width=True)


def render_clusters(r):
    fig = _scatter3d_clusters(r["obstacle_points"], r["cluster_labels"])
    st.plotly_chart(fig, use_container_width=True)


def render_3d(r):
    color_mode = st.radio(
        "Color by", ["Ground / Obstacle", "Cluster Labels"],
        horizontal=True,
    )

    if color_mode == "Ground / Obstacle":
        fig = _scatter3d_ground_obstacle(r["ground_points"], r["obstacle_points"])
    else:
        fig = _scatter3d_clusters(r["obstacle_points"], r["cluster_labels"])
        ground = r["ground_points"]
        if len(ground) > 0:
            fig.add_trace(go.Scatter3d(
                x=ground[:, 0], y=ground[:, 1], z=ground[:, 2],
                mode="markers",
                marker=dict(size=1, color="blue", opacity=0.2),
                name=f"Ground ({len(ground):,})",
            ))

    st.plotly_chart(fig, use_container_width=True)


def main():
    st.set_page_config(page_title="LiDAR Pipeline", layout="wide")
    st.subheader("LiDAR Visualization")

    SAMPLES = {
        "Sample 1": {"points": "data/sample1.txt", "image": "data/sample1.png"},
        "Sample 2": {"points": "data/sample2.txt", "image": "data/sample2.png"},
        "Sample 3": {"points": "data/sample3.txt", "image": "data/sample3.png"},
    }

    with st.sidebar:
        st.header("Input")
        sample_name = st.selectbox("Point cloud", list(SAMPLES.keys()))
        input_file = SAMPLES[sample_name]["points"]
        image_file = SAMPLES[sample_name]["image"]

        voxel_size = 0.1
        radius_scale = 0.03
        min_neighbors = 5
        ransac_iters = 100
        dist_thresh = 0.2
        normal_thresh = 0.9
        eps = 0.5
        min_samples = 10
        min_cluster = 0
        max_cluster = 0

        with st.popover("Preprocessing", use_container_width=True):
            voxel_size = st.slider(
                "Voxel size (larger = fewer points, faster)",
                0.01, 1.0, 0.1, 0.01,
            )
            radius_scale = st.slider(
                "Outlier radius scale (larger = keeps more points)",
                0.01, 0.20, 0.03, 0.005,
            )
            min_neighbors = st.slider(
                "Min neighbors (higher = stricter filtering)",
                1, 20, 5,
            )

        with st.popover("RANSAC", use_container_width=True):
            ransac_iters = st.slider(
                "Iterations (more = better fit, slower)",
                10, 500, 100, 10,
            )
            dist_thresh = st.slider(
                "Distance threshold (larger = thicker ground layer)",
                0.05, 1.0, 0.2, 0.05,
            )
            normal_thresh = st.slider(
                "Normal threshold (lower = allows tilted planes)",
                0.5, 1.0, 0.9, 0.05,
            )

        with st.popover("Clustering", use_container_width=True):
            eps = st.slider(
                "DBSCAN eps (larger = merges nearby objects)",
                0.1, 3.0, 0.5, 0.1,
            )
            min_samples = st.slider(
                "Min samples (higher = ignores small groups)",
                3, 50, 10,
            )
            min_cluster = st.number_input(
                "Min cluster size, 0=off (filters tiny clusters)",
                0, 1000, 0,
            )
            max_cluster = st.number_input(
                "Max cluster size, 0=off (filters huge clusters)",
                0, 50000, 0,
            )

        run_button = st.button("Run Pipeline", type="primary", use_container_width=True)

    if run_button and input_file:
        path = Path(input_file)
        if not path.exists():
            st.error(f"File not found: {path}")
            return

        with st.spinner("Running pipeline..."):
            if path.suffix == ".txt":
                raw_points = load_kitti_txt(str(path))
            else:
                raw_points = load_kitti_bin(str(path))

            downsampled = voxel_downsample(raw_points, voxel_size=voxel_size)

            filtered = radial_outlier_removal(
                downsampled, radius_scale=radius_scale, min_neighbors=min_neighbors
            )

            plane, ground_mask = ransac_ground_plane(
                filtered,
                num_iterations=ransac_iters,
                distance_threshold=dist_thresh,
                normal_threshold=normal_thresh,
            )
            ground_points = filtered[ground_mask]
            obstacle_points = filtered[~ground_mask]

            cluster_result = dbscan_cluster(
                obstacle_points,
                eps=eps,
                min_samples=min_samples,
                min_cluster_size=min_cluster if min_cluster > 0 else None,
                max_cluster_size=max_cluster if max_cluster > 0 else None,
            )

            st.session_state["result"] = {
                "image_file": image_file,
                "raw_count": len(raw_points),
                "downsampled_points": downsampled,
                "downsampled_count": len(downsampled),
                "filtered_points": filtered,
                "filtered_count": len(filtered),
                "outliers_removed": len(downsampled) - len(filtered),
                "outlier_pct": (len(downsampled) - len(filtered)) / max(len(downsampled), 1) * 100,
                "plane_model": plane,
                "plane_equation": plane.equation_string,
                "ground_points": ground_points,
                "ground_count": len(ground_points),
                "obstacle_points": obstacle_points,
                "obstacle_count": len(obstacle_points),
                "cluster_labels": cluster_result.labels,
                "num_clusters": cluster_result.num_clusters,
                "cluster_sizes": cluster_result.cluster_sizes,
                "noise_count": cluster_result.noise_count,
            }

    if "result" not in st.session_state:
        st.info("Configure parameters in the sidebar and click **Run Pipeline** to begin.")
        return

    r = st.session_state["result"]

    tab_preproc, tab_ground, tab_cluster, tab_3d = st.tabs(
        ["Preprocessing", "Ground Segmentation", "Clusters", "Full 3D View"]
    )

    with tab_preproc:
        render_preprocessing(r)
    with tab_ground:
        render_ground(r)
    with tab_cluster:
        render_clusters(r)
    with tab_3d:
        render_3d(r)


if __name__ == "__main__":
    main()
