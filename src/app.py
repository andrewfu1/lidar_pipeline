"""Streamlit UI for LiDAR pipeline visualization - This part is vibecoded"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import numpy as np
import plotly.graph_objects as go

from src.pipeline import PipelineParams, FrameResult, run_frame_pipeline, run_sequence_pipeline
from src.visualizations import (
    scatter_2d,
    scatter_3d_ground_obstacle,
    scatter_3d_clusters,
    bev_with_tracks,
)

SAMPLES = {
    "Sample 1 (start of drive)": {"points": "data/sample1.txt", "image": "data/sample1.png"},
    "Sample 2 (middle of drive)": {"points": "data/sample2.txt", "image": "data/sample2.png"},
    "Sample 3 (end of drive)": {"points": "data/sample3.txt", "image": "data/sample3.png"},
}


def get_params_from_sidebar() -> PipelineParams:
    """Render parameter controls in sidebar and return PipelineParams."""
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

    return PipelineParams(
        voxel_size=voxel_size,
        radius_scale=radius_scale,
        min_neighbors=min_neighbors,
        ransac_iters=ransac_iters,
        dist_thresh=dist_thresh,
        normal_thresh=normal_thresh,
        eps=eps,
        min_samples=min_samples,
        min_cluster=min_cluster,
        max_cluster=max_cluster,
    )


# =============================================================================
# Single Frame Mode
# =============================================================================

def render_preprocessing_tab(r: FrameResult):
    """Render preprocessing tab content."""
    if r.image_path:
        st.image(r.image_path, caption="Left color camera (image_02)", use_container_width=True)

    st.caption(
        f"Downsampled: **{r.downsampled_count:,}** pts | "
        f"Filtered: **{r.filtered_count:,}** pts | "
        f"Removed: **{r.outliers_removed:,}** outliers"
    )

    col_left, col_right = st.columns(2)
    with col_left:
        fig = scatter_2d(
            [r.downsampled_points],
            ["Downsampled"],
            ["#4363d8"],
            "Before Outlier Removal",
            "X (m)", "Y (m)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        fig = scatter_2d(
            [r.filtered_points],
            ["Filtered"],
            ["#3cb44b"],
            "After Outlier Removal",
            "X (m)", "Y (m)",
        )
        st.plotly_chart(fig, use_container_width=True)


def render_ground_tab(r: FrameResult):
    """Render ground segmentation tab content."""
    view = st.radio("View", ["Bird's Eye (XY)", "Side View (XZ)"], horizontal=True)

    ground = r.ground_points
    obstacle = r.obstacle_points

    if view == "Bird's Eye (XY)":
        fig = scatter_2d(
            [ground, obstacle],
            [f"Ground ({len(ground):,})", f"Obstacles ({len(obstacle):,})"],
            ["blue", "red"],
            "Ground vs Obstacles - Bird's Eye",
            "X (m)", "Y (m)",
        )
    else:
        fig = scatter_2d(
            [ground, obstacle],
            [f"Ground ({len(ground):,})", f"Obstacles ({len(obstacle):,})"],
            ["blue", "red"],
            "Ground vs Obstacles - Side View",
            "X (m)", "Z (m)",
            x_idx=0, y_idx=2,
        )
        plane = r.plane_model
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


def render_clusters_tab(r: FrameResult):
    """Render clusters tab content."""
    fig = scatter_3d_clusters(r.obstacle_points, r.cluster_labels)
    st.plotly_chart(fig, use_container_width=True)


def render_3d_tab(r: FrameResult):
    """Render full 3D view tab content."""
    color_mode = st.radio(
        "Color by", ["Ground / Obstacle", "Cluster Labels"],
        horizontal=True,
    )

    if color_mode == "Ground / Obstacle":
        fig = scatter_3d_ground_obstacle(r.ground_points, r.obstacle_points)
    else:
        fig = scatter_3d_clusters(r.obstacle_points, r.cluster_labels)
        ground = r.ground_points
        if len(ground) > 0:
            fig.add_trace(go.Scatter3d(
                x=ground[:, 0], y=ground[:, 1], z=ground[:, 2],
                mode="markers",
                marker=dict(size=1, color="blue", opacity=0.2),
                name=f"Ground ({len(ground):,})",
            ))

    st.plotly_chart(fig, use_container_width=True)


def render_single_frame_mode():
    """Render the single frame analysis mode."""
    with st.sidebar:
        st.header("Input")
        sample_name = st.selectbox("Point cloud", list(SAMPLES.keys()))
        input_file = SAMPLES[sample_name]["points"]
        image_file = SAMPLES[sample_name]["image"]

        st.header("Parameters")
        params = get_params_from_sidebar()

        run_button = st.button("Run Frame", type="primary", use_container_width=True)

    # Run pipeline
    if run_button:
        path = Path(input_file)
        if not path.exists():
            st.error(f"File not found: {path}")
            return

        with st.spinner("Running pipeline..."):
            result = run_frame_pipeline(
                points_path=str(path),
                params=params,
                image_path=image_file,
            )
            st.session_state["single_frame_result"] = result

    # Display results
    if "single_frame_result" not in st.session_state:
        st.info("Configure parameters in the sidebar and click **Run Pipeline** to begin.")
        return

    r = st.session_state["single_frame_result"]

    tab_preproc, tab_ground, tab_cluster, tab_3d = st.tabs(
        ["Preprocessing", "Ground Segmentation", "Clusters", "Full 3D View"]
    )

    with tab_preproc:
        render_preprocessing_tab(r)
    with tab_ground:
        render_ground_tab(r)
    with tab_cluster:
        render_clusters_tab(r)
    with tab_3d:
        render_3d_tab(r)


# =============================================================================
# Sequence Mode
# =============================================================================

def render_playback_controls(total_frames: int):
    """Render frame navigation and playback controls."""
    # Initialize frame_idx if not present
    if "frame_idx" not in st.session_state:
        st.session_state["frame_idx"] = 0

    # Handle frame navigation before widgets render
    current_idx = st.session_state["frame_idx"]

    if st.session_state.pop("_advance_frame", False):
        if current_idx < total_frames - 1:
            st.session_state["frame_idx"] = current_idx + 1
        else:
            st.session_state["auto_play"] = False
    if st.session_state.pop("_prev_frame", False) and current_idx > 0:
        st.session_state["frame_idx"] = current_idx - 1
    if st.session_state.pop("_next_frame", False) and current_idx < total_frames - 1:
        st.session_state["frame_idx"] = current_idx + 1

    # Frame slider - don't pass default value since we manage state ourselves
    frame_idx = st.slider(
        "Frame",
        0, total_frames - 1,
        key="frame_idx",
    )

    # Navigation buttons
    col_prev, col_play, col_next, col_speed = st.columns([1, 1, 1, 1])

    with col_prev:
        if st.button("< Prev", use_container_width=True, disabled=frame_idx == 0):
            st.session_state["_prev_frame"] = True
            st.rerun()

    with col_play:
        is_playing = st.session_state.get("auto_play", False)
        play_label = "Stop" if is_playing else "Play"
        if st.button(play_label, use_container_width=True):
            st.session_state["auto_play"] = not is_playing
            st.rerun()

    with col_next:
        if st.button("Next >", use_container_width=True, disabled=frame_idx >= total_frames - 1):
            st.session_state["_next_frame"] = True
            st.rerun()

    with col_speed:
        fps = st.selectbox("FPS", [1, 3], index=1)

    return frame_idx, fps


def render_sequence_mode():
    """Render the sequence tracking mode."""
    with st.sidebar:
        st.header("Sequence")
        seq_dir = st.text_input(
            "Directory path",
            value="data/2011_09_26_drive_0001_extract",
        )

        st.header("Parameters")
        params = get_params_from_sidebar()

        process_button = st.button("Run Sequence", type="primary", use_container_width=True)

    # Process sequence
    if process_button:
        if not Path(seq_dir).exists():
            st.error(f"Directory not found: {seq_dir}")
            return

        progress_bar = st.progress(0, text="Processing frames...")

        def progress_callback(current: int, total: int):
            if total > 0:
                progress_bar.progress(current / total, text=f"Processing frame {current + 1}/{total}...")

        results = run_sequence_pipeline(
            seq_dir=seq_dir,
            params=params,
            progress_callback=progress_callback,
        )

        progress_bar.empty()

        if not results:
            st.error("No frames found in sequence directory.")
            return

        st.session_state["sequence_results"] = results
        st.session_state["sequence_processed"] = True
        st.session_state["frame_idx"] = 0
        st.session_state["auto_play"] = False
        st.rerun()

    # Check if sequence is processed
    if not st.session_state.get("sequence_processed"):
        st.info("Enter a sequence directory path and click **Process Sequence** to begin.")
        return

    results = st.session_state["sequence_results"]
    total_frames = len(results)

    # Playback controls
    frame_idx, fps = render_playback_controls(total_frames)

    # Get current frame result
    r = results[frame_idx]

    # Stats row
    col1, col2, col3 = st.columns(3)
    col1.metric("Frame", f"{frame_idx + 1} / {total_frames}")
    col2.metric("Clusters", r.num_clusters)
    col3.metric("Tracks", len(r.tracks))

    # Bird's eye view with tracks - use container for smoother updates
    chart_container = st.container()
    with chart_container:
        fig = bev_with_tracks(r.obstacle_points, r.tracks)
        st.plotly_chart(fig, use_container_width=True, key=f"bev_{frame_idx}")

    # Auto-play loop
    if st.session_state.get("auto_play"):
        time.sleep(1.0 / fps)
        st.session_state["_advance_frame"] = True
        st.rerun()


# =============================================================================
# Main
# =============================================================================

def main():
    st.set_page_config(page_title="LiDAR Pipeline", layout="wide")
    st.title("LiDAR Pipeline")

    # Mode toggle at top
    col1, col2 = st.columns(2)

    current_mode = st.session_state.get("mode", "single")

    with col1:
        single_selected = current_mode == "single"
        if st.button(
            "Single Frame",
            use_container_width=True,
            type="primary" if single_selected else "secondary",
        ):
            st.session_state["mode"] = "single"
            st.rerun()

    with col2:
        seq_selected = current_mode == "sequence"
        if st.button(
            "Sequence (Tracking)",
            use_container_width=True,
            type="primary" if seq_selected else "secondary",
        ):
            st.session_state["mode"] = "sequence"
            st.rerun()

    st.divider()

    # Route to appropriate mode
    if current_mode == "sequence":
        render_sequence_mode()
    else:
        render_single_frame_mode()


if __name__ == "__main__":
    main()
