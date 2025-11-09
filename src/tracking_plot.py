# viz_topdown.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from lidar_types import Scene


def plot_scene_pair(
    scene_a: Scene,
    scene_b: Scene,
    pairs: List[Tuple[int, int]],
):
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Tracking: current → next scene")

    # Plot raw points (optional, light gray)
    if scene_a.points is not None:
        plt.scatter(
            scene_a.points[:, 0],
            scene_a.points[:, 1],
            s=1,
            c="lightgray",
            alpha=0.5,
            label="scene A points",
        )
    if scene_b.points is not None:
        plt.scatter(
            scene_b.points[:, 0],
            scene_b.points[:, 1],
            s=1,
            c="lightblue",
            alpha=0.5,
            label="scene B points",
        )

    # Cluster centroids
    clusters_a = scene_a.scene_clusters or []
    clusters_b = scene_b.scene_clusters or []

    centroids_a = (
        np.array([c.geometry.centroid[:2] for c in clusters_a])
        if clusters_a
        else np.empty((0, 2))
    )
    centroids_b = (
        np.array([c.geometry.centroid[:2] for c in clusters_b])
        if clusters_b
        else np.empty((0, 2))
    )

    if len(centroids_a):
        plt.scatter(
            centroids_a[:, 0],
            centroids_a[:, 1],
            c="red",
            s=40,
            label="current centroids",
        )
    if len(centroids_b):
        plt.scatter(
            centroids_b[:, 0],
            centroids_b[:, 1],
            c="green",
            s=40,
            label="next centroids",
        )

    # Draw match lines (track connections)
    for i, j in pairs:
        if i < len(centroids_a) and j < len(centroids_b):
            xa, ya = centroids_a[i]
            xb, yb = centroids_b[j]
            plt.plot([xa, xb], [ya, yb], c="k", linewidth=1)

    plt.legend(loc="upper right", fontsize=8)
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig("./data/tracking_plot.pdf")
    plt.close()
