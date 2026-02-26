import matplotlib.pyplot as plt
import json
import os
import math
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
def plot_2d_points_subplot(ax, points, title):
    lose_points = np.array(points[:-1])  # losing points
    win_point = points[-1]  # last point = win
    colors = ["blue", "green", "purple"]
    
    if len(lose_points) >= 3:
        kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(lose_points)
        
        for cluster_id in range(3):
            cluster_pts = lose_points[labels == cluster_id]
            
            ax.scatter(cluster_pts[:, 0], cluster_pts[:, 1], s=30, color=colors[cluster_id], alpha=0.7)
            
            if len(cluster_pts) >= 3:
                hull = ConvexHull(cluster_pts)
                hull_pts = cluster_pts[hull.vertices]
                hull_pts = np.vstack([hull_pts, hull_pts[0]])  # close loop
                
                ax.plot(hull_pts[:, 0], hull_pts[:, 1], color=colors[cluster_id], linewidth=2)
    else:
        ax.scatter(lose_points[:, 0], lose_points[:, 1], s=30, color="blue")
    
    ax.scatter(win_point[0], win_point[1], s=90, color="red", edgecolor="black", zorder=5)
    ax.axhline(0, linewidth=0.5)
    ax.axvline(0, linewidth=0.5)
    ax.set_title(title)
    ax.grid(True)


img_indices = range(400, 500)  # ANY number of indices

json_path1 = "/workspace/data/Pulkit/pref_qual/asthetic.json"
json_path2 = "/workspace/data/Pulkit/pref_qual/imagereward.json"
base_path = "/workspace/data/Pulkit/pref_qual/dataset_full"

with open(json_path1, "r") as f:
    all_results1 = json.load(f)

with open(json_path2, "r") as f:
    all_results2 = json.load(f)

n = len(img_indices)
cols = min(4, n)
rows = math.ceil(n / cols)
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
axes = axes.flatten() if n > 1 else [axes]

for ax, img_ind in zip(axes, img_indices):
    pt_list = []
    
    for i in range(1, 100):
        path = f"{base_path}/lose{i}/{img_ind}.png"
        if path in all_results1 and path in all_results2:
            pt_list.append((all_results1[path], all_results2[path]))
    
    win_path = f"{base_path}/win/{img_ind}.png"
    if win_path in all_results1 and win_path in all_results2:
        pt_list.append((all_results1[win_path], all_results2[win_path]))
    
    plot_2d_points_subplot(ax, pt_list, title=f"Index {img_ind}")

for ax in axes[len(img_indices):]:
    ax.axis("off")

plt.tight_layout()
plt.savefig("/workspace/data/Pulkit/pref_qual/plot.png", dpi=200)
plt.show()
