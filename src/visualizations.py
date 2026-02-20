"""Plotly visualization functions for LiDAR pipeline - This part is vibecoded"""

import numpy as np
import plotly.graph_objects as go

CLUSTER_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9A6324", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#ffffff",
]


def scatter_2d(points_list, names, colors, title, xlabel, ylabel, x_idx=0, y_idx=1):
    """Create a 2D scatter plot with multiple point sets."""
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


def scatter_3d_ground_obstacle(ground, obstacle):
    """Create a 3D scatter plot showing ground vs obstacle points."""
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


def scatter_3d_clusters(obstacle_points, labels):
    """Create a 3D scatter plot with colored clusters."""
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


def bev_with_tracks(obstacle_points, tracks):
    """
    Bird's eye view with ego vehicle centered (radar/surveillance style).

    KITTI coordinate system: X = forward, Y = left, Z = up
    We plot: horizontal = Y (left-right), vertical = X (forward)

    Args:
        obstacle_points: Nx3 array of obstacle points
        tracks: List of Track objects with bounding_box and track_id attributes

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    # Plot obstacle points - swap axes for driving view (Y horizontal, X vertical)
    if len(obstacle_points) > 0:
        fig.add_trace(go.Scattergl(
            x=obstacle_points[:, 1],  # Y = left-right
            y=obstacle_points[:, 0],  # X = forward
            mode="markers",
            marker=dict(size=2, color="#666666", opacity=0.5),
            name="Points",
            hoverinfo="skip",
        ))

    # Draw ego vehicle marker at origin (triangle pointing up)
    fig.add_trace(go.Scatter(
        x=[0, -1.0, 1.0, 0],
        y=[0.5, -1.5, -1.5, 0.5],
        mode="lines",
        fill="toself",
        fillcolor="rgba(0, 150, 255, 0.8)",
        line=dict(color="rgb(0, 150, 255)", width=2),
        name="Ego Vehicle",
        showlegend=False,
        hoverinfo="skip",
    ))

    # Draw bounding boxes for each track - show ID on hover only
    for track in tracks:
        bbox = track.bounding_box
        color = CLUSTER_COLORS[track.track_id % len(CLUSTER_COLORS)]

        # Convert hex color to rgba with transparency
        if color.startswith("#"):
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            fill_color = f"rgba({r},{g},{b},0.15)"
        else:
            fill_color = color

        # Swap axes: plot Y on horizontal, X on vertical
        fig.add_trace(go.Scatter(
            x=[bbox.min_y, bbox.max_y, bbox.max_y, bbox.min_y, bbox.min_y],
            y=[bbox.min_x, bbox.min_x, bbox.max_x, bbox.max_x, bbox.min_x],
            mode="lines",
            line=dict(color=color, width=2),
            fill="toself",
            fillcolor=fill_color,
            name=f"Track {track.track_id}",
            showlegend=False,
            hovertemplate=f"Track {track.track_id}<extra></extra>",
        ))

    fig.update_layout(
        title=None,
        xaxis=dict(
            title="← Left (m)    Right (m) →",
            range=[-40, 40],
            showgrid=True,
            gridcolor="rgba(100,100,100,0.3)",
            zeroline=True,
            zerolinecolor="rgba(150,150,150,0.5)",
        ),
        yaxis=dict(
            title="Forward (m) →",
            range=[-40, 40],
            showgrid=True,
            gridcolor="rgba(100,100,100,0.3)",
            scaleanchor="x",
            zeroline=True,
            zerolinecolor="rgba(150,150,150,0.5)",
        ),
        height=650,
        margin=dict(l=50, r=20, t=20, b=50),
        showlegend=False,
        plot_bgcolor="rgb(20, 20, 25)",
        paper_bgcolor="rgb(20, 20, 25)",
        font=dict(color="white"),
        hoverlabel=dict(bgcolor="rgba(0,0,0,0.8)", font_size=14),
    )

    return fig
