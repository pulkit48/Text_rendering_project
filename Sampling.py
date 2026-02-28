import matplotlib.pyplot as plt
import json
import numpy as np
from scipy.spatial import ConvexHull


def visualize_all_configs(base_path,
                          json_path1,
                          json_path2,
                          output_prefix,
                          img_index,
                          configs,
                          epsilon=0.05):

    # ----------------------------
    # Load feature files
    # ----------------------------
    with open(json_path1) as f:
        all_results1 = json.load(f)

    with open(json_path2) as f:
        all_results2 = json.load(f)

    # ----------------------------
    # Load sampling outputs
    # ----------------------------
    sampled_data = {}
    for cfg in configs:
        with open(f"{output_prefix}_{cfg}.json") as f:
            sampled_data[cfg] = json.load(f)

    # Load base clustering info for hull recovery
    with open(f"{output_prefix}_kmeans.json") as f:
        kmeans_data = json.load(f)

    with open(f"{output_prefix}_epsilon_kmeans.json") as f:
        epsilon_kmeans_data = json.load(f)

    # ----------------------------
    # Collect original lose points
    # ----------------------------
    original_points = []
    for i in range(1, 100):
        path = f"{base_path}/lose{i}/{img_index}.png"
        if path in all_results1 and path in all_results2:
            original_points.append(
                (all_results1[path], all_results2[path])
            )

    original_points = np.array(original_points)

    win_path = f"{base_path}/win/{img_index}.png"
    win_point = np.array([
        all_results1[win_path],
        all_results2[win_path]
    ])

    # ----------------------------
    # Common axis bounds
    # ----------------------------
    all_x = list(original_points[:, 0]) + [win_point[0]]
    all_y = list(original_points[:, 1]) + [win_point[1]]

    xmin = min(all_x + [win_point[0] - epsilon])
    xmax = max(all_x + [win_point[0] + epsilon])
    ymin = min(all_y + [win_point[1] - epsilon])
    ymax = max(all_y + [win_point[1] + epsilon])

    pad_x = 0.05 * (xmax - xmin)
    pad_y = 0.05 * (ymax - ymin)

    xmin -= pad_x
    xmax += pad_x
    ymin -= pad_y
    ymax += pad_y

    # ----------------------------
    # 2 × 3 Grid
    # ----------------------------
    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    axes = axes.flatten()

    cluster_colors = ["blue", "green", "purple"]

    for ax, cfg in zip(axes, configs):

        # ----------------------------------
        # Plot base points (with epsilon split)
        # ----------------------------------
        if "epsilon" in cfg:

            distances = np.linalg.norm(original_points - win_point, axis=1)
            outside_mask = distances > epsilon

            inside_pts = original_points[~outside_mask]
            outside_pts = original_points[outside_mask]

            if len(inside_pts) > 0:
                ax.scatter(inside_pts[:, 0],
                           inside_pts[:, 1],
                           s=20,
                           color="lightgray",
                           alpha=0.6)

            if len(outside_pts) > 0:
                ax.scatter(outside_pts[:, 0],
                           outside_pts[:, 1],
                           s=25,
                           color="gray",
                           alpha=0.6)

            circle = plt.Circle(win_point,
                                epsilon,
                                color="red",
                                fill=False,
                                linestyle="--",
                                linewidth=2)
            ax.add_patch(circle)

        else:
            ax.scatter(original_points[:, 0],
                       original_points[:, 1],
                       s=25,
                       color="gray",
                       alpha=0.6)

        cfg_output = sampled_data[cfg].get(str(img_index), [])

        # ----------------------------------
        # NON-CLUSTERED CONFIG
        # ----------------------------------
        if isinstance(cfg_output, list):

            selected_pts = [
                (all_results1[p], all_results2[p])
                for p in cfg_output
            ]

            if len(selected_pts) > 0:
                selected_pts = np.array(selected_pts)

                ax.scatter(selected_pts[:, 0],
                           selected_pts[:, 1],
                           s=80,
                           color="orange",
                           edgecolor="black",
                           zorder=5)

        # ----------------------------------
        # CLUSTERED CONFIG
        # ----------------------------------
        elif isinstance(cfg_output, dict):

            # Determine correct hull source
            if cfg.startswith("epsilon"):
                hull_source = epsilon_kmeans_data
            else:
                hull_source = kmeans_data

            full_clusters = hull_source[str(img_index)]

            for idx, cluster_name in enumerate(full_clusters):

                # Full cluster for hull
                full_paths = full_clusters[cluster_name]
                full_pts = [
                    (all_results1[p], all_results2[p])
                    for p in full_paths
                ]

                if len(full_pts) >= 3:
                    full_pts = np.array(full_pts)
                    hull = ConvexHull(full_pts)
                    hull_pts = full_pts[hull.vertices]
                    hull_pts = np.vstack([hull_pts, hull_pts[0]])

                    ax.plot(hull_pts[:, 0],
                            hull_pts[:, 1],
                            color=cluster_colors[idx % 3],
                            linewidth=2)

                # Selected subset (diversity)
                selected_paths = cfg_output.get(cluster_name, [])
                selected_pts = [
                    (all_results1[p], all_results2[p])
                    for p in selected_paths
                ]

                if len(selected_pts) > 0:
                    selected_pts = np.array(selected_pts)

                    ax.scatter(selected_pts[:, 0],
                               selected_pts[:, 1],
                               s=70,
                               color=cluster_colors[idx % 3],
                               alpha=0.5,
                               edgecolor="black",
                               zorder=5)

        # ----------------------------------
        # Win point
        # ----------------------------------
        ax.scatter(win_point[0],
                   win_point[1],
                   s=130,
                   color="red",
                   edgecolor="black",
                   zorder=6)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_title(cfg)
        ax.grid(True)

    # Hide unused axes if fewer than 6 configs
    for ax in axes[len(configs):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# ----------------------------
# CALL
# ----------------------------

configs = [
    "none",
    "epsilon",
    "kmeans",
    "epsilon_kmeans",
    "kmeans_diversity",
    "epsilon_kmeans_diversity",
]

visualize_all_configs(
    base_path="/workspace/data/Pulkit/pref_qual/dataset_full",
    json_path1="/workspace/data/Pulkit/pref_qual/asthetic.json",
    json_path2="/workspace/data/Pulkit/pref_qual/imagereward.json",
    output_prefix="/workspace/data/Pulkit/pref_qual/output/output",
    img_index=400,
    configs=configs,
    epsilon=0.05
)
