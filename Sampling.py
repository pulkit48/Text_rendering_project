import json
import os
import numpy as np
from sklearn.cluster import KMeans


def find_k_clique(graph, k):

    def backtrack(clique, candidates):
        if len(clique) == k:
            return clique

        if len(clique) + len(candidates) < k:
            return None

        for i, v in enumerate(candidates):
            new_candidates = [
                u for u in candidates[i+1:] if u in graph[v]
            ]
            result = backtrack(clique + [v], new_candidates)
            if result is not None:
                return result

        return None

    return backtrack([], list(graph.keys()))


def max_min_dispersion_binary_clique(points, k):

    points = np.array(points)
    n = len(points)

    if n <= k:
        return list(range(n))

    D = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    distances = sorted(set(D[i, j] for i in range(n) for j in range(i+1, n)))

    low, high = 0, len(distances) - 1
    best_subset = None

    while low <= high:
        mid = (low + high) // 2
        t = distances[mid]

        graph = {i: set() for i in range(n)}

        for i in range(n):
            for j in range(i+1, n):
                if D[i, j] >= t:
                    graph[i].add(j)
                    graph[j].add(i)

        clique = find_k_clique(graph, k)

        if clique is not None:
            best_subset = clique
            low = mid + 1
        else:
            high = mid - 1

    if best_subset is None:
        return list(range(min(k, n)))

    return best_subset


def filter_by_epsilon(features, win_feature, epsilon):
    distances = np.linalg.norm(features - win_feature, axis=1)
    return distances > epsilon


def adaptive_sample_from_clusters(cluster_feats, cluster_paths, total_k=16):
    """
    Adaptive sampling: handles cases where clusters have fewer points than needed.
    Redistributes the deficit to larger clusters.
    """
    
    n_clusters = len(cluster_feats)
    
    if n_clusters == 0:
        return []
    
    # Sort clusters by size (largest first)
    cluster_items = [(cid, cluster_feats[cid], cluster_paths[cid]) 
                     for cid in cluster_feats.keys()]
    cluster_items.sort(key=lambda x: len(x[1]), reverse=True)
    
    # Initial allocation: 6 for largest, 5 for others
    allocation = {}
    remaining = total_k
    
    # First pass: assign base allocation (5-5-6)
    for i, (cid, feats, _) in enumerate(cluster_items):
        base = 6 if i == 0 else 5
        take = min(base, len(feats), remaining)
        allocation[cid] = take
        remaining -= take
    
    # Second pass: distribute remaining to largest clusters
    for cid, feats, _ in cluster_items:
        if remaining <= 0:
            break
        
        current = allocation[cid]
        max_more = len(feats) - current
        give = min(remaining, max_more)
        allocation[cid] += give
        remaining -= give
    
    # Sample from each cluster
    all_selected_paths = []
    
    for cid in cluster_feats.keys():
        n_select = allocation.get(cid, 0)
        
        if n_select > 0 and len(cluster_feats[cid]) > 0:
            feats = np.array(cluster_feats[cid])
            paths = np.array(cluster_paths[cid])
            
            selected_indices = max_min_dispersion_binary_clique(feats, n_select)
            all_selected_paths.extend(paths[selected_indices].tolist())
    
    return all_selected_paths


def process_single_config(config_name,
                         features,
                         paths,
                         win_feature,
                         epsilon=0.05,
                         total_k=16):
    """
    Process a single configuration and return the result.
    
    Returns:
        - For non-clustered: list of paths
        - For clustered: dict with cluster_0, cluster_1, cluster_2 keys
    """
    
    result = None
    
    # ============================================================
    # 1. none - No processing
    # ============================================================
    if config_name == "none":
        result = paths.tolist()
    
    # ============================================================
    # 2. epsilon_only - Only epsilon filtering
    # ============================================================
    elif config_name == "epsilon_only":
        mask = filter_by_epsilon(features, win_feature, epsilon)
        result = paths[mask].tolist()
    
    # ============================================================
    # 3. kmeans_only - Only clustering
    # ============================================================
    elif config_name == "kmeans_only":
        if len(features) < 3:
            return None
        
        labels = KMeans(n_clusters=3, random_state=42, n_init="auto").fit_predict(features)
        
        result = {"cluster_0": [], "cluster_1": [], "cluster_2": []}
        for p, l in zip(paths, labels):
            result[f"cluster_{l}"].append(p)
    
    # ============================================================
    # 4. diversity_only - Only max diversity sampling (16 points)
    # ============================================================
    elif config_name == "diversity_only":
        if len(features) == 0:
            return None
        
        if len(features) < total_k:
            selected_indices = list(range(len(features)))
        else:
            selected_indices = max_min_dispersion_binary_clique(features, total_k)
        result = paths[selected_indices].tolist()
    
    # ============================================================
    # 5. epsilon_kmeans - Epsilon + KMeans
    # ============================================================
    elif config_name == "epsilon_kmeans":
        mask = filter_by_epsilon(features, win_feature, epsilon)
        filtered_features = features[mask]
        filtered_paths = paths[mask]
        
        if len(filtered_features) < 3:
            return None
        
        labels = KMeans(n_clusters=3, random_state=42, n_init="auto").fit_predict(filtered_features)
        
        result = {"cluster_0": [], "cluster_1": [], "cluster_2": []}
        for p, l in zip(filtered_paths, labels):
            result[f"cluster_{l}"].append(p)
    
    # ============================================================
    # 6. epsilon_diversity - Epsilon + Diversity
    # ============================================================
    elif config_name == "epsilon_diversity":
        mask = filter_by_epsilon(features, win_feature, epsilon)
        filtered_features = features[mask]
        filtered_paths = paths[mask]
        
        if len(filtered_features) == 0:
            return None
        
        if len(filtered_features) < total_k:
            selected_indices = list(range(len(filtered_features)))
        else:
            selected_indices = max_min_dispersion_binary_clique(filtered_features, total_k)
        
        result = filtered_paths[selected_indices].tolist()
    
    # ============================================================
    # 7. kmeans_diversity - KMeans + Diversity (5-5-6 adaptive)
    # ============================================================
    elif config_name == "kmeans_diversity":
        if len(features) < 3:
            return None
        
        labels = KMeans(n_clusters=3, random_state=42, n_init="auto").fit_predict(features)
        
        cluster_feats = {0: [], 1: [], 2: []}
        cluster_paths = {0: [], 1: [], 2: []}
        
        for p, f, l in zip(paths, features, labels):
            cluster_paths[l].append(p)
            cluster_feats[l].append(f)
        
        result = adaptive_sample_from_clusters(cluster_feats, cluster_paths, total_k)
    
    # ============================================================
    # 8. epsilon_kmeans_diversity - Full pipeline
    # ============================================================
    elif config_name == "epsilon_kmeans_diversity":
        mask = filter_by_epsilon(features, win_feature, epsilon)
        filtered_features = features[mask]
        filtered_paths = paths[mask]
        
        if len(filtered_features) < 3:
            return None
        
        labels = KMeans(n_clusters=3, random_state=42, n_init="auto").fit_predict(filtered_features)
        
        cluster_feats = {0: [], 1: [], 2: []}
        cluster_paths = {0: [], 1: [], 2: []}
        
        for p, f, l in zip(filtered_paths, filtered_features, labels):
            cluster_paths[l].append(p)
            cluster_feats[l].append(f)
        
        result = adaptive_sample_from_clusters(cluster_feats, cluster_paths, total_k)
    
    return result


def create_output_structure(output_dir, configs):
    """
    Create the output directory structure for all configurations.
    """
    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Created main output directory: {output_dir}")
    
    # Configs that need folder structure (with clustering)
    clustered_configs = ["kmeans_only", "epsilon_kmeans"]
    
    for config in configs:
        if config in clustered_configs:
            config_dir = os.path.join(output_dir, config)
            os.makedirs(config_dir, exist_ok=True)
            print(f"📁 Created directory for {config}: {config_dir}")
    
    print("✅ Output structure created successfully!\n")


def main(json_path1,
         json_path2,
         base_path,
         output_dir,
         epsilon=0.05,
         total_k=16,
         img_indices=range(400, 405)):
    
    # All configurations
    configs = [
        "none",
        "epsilon_only",
        "kmeans_only",
        "diversity_only",
        "epsilon_kmeans",
        "epsilon_diversity",
        "kmeans_diversity",
        "epsilon_kmeans_diversity"
    ]
    
    # Create output directory structure
    create_output_structure(output_dir, configs)
    
    # Load data
    print("📂 Loading JSON data...")
    with open(json_path1, "r") as f:
        all_results1 = json.load(f)
    
    with open(json_path2, "r") as f:
        all_results2 = json.load(f)
    print("✅ JSON data loaded successfully!\n")
    
    # Initialize output storage for each config
    config_outputs = {config: {} for config in configs}
    
    # Process each image index
    print(f"🔄 Processing {len(list(img_indices))} image indices...")
    for img_ind in img_indices:
        
        try:
            # Collect all losing points
            paths = []
            features = []
            
            for i in range(1, 100):
                path = f"{base_path}/lose{i}/{img_ind}.png"
                if path in all_results1 and path in all_results2:
                    paths.append(path)
                    features.append([all_results1[path], all_results2[path]])
            
            # Get winning point
            win_path = f"{base_path}/win/{img_ind}.png"
            
            if win_path not in all_results1 or win_path not in all_results2:
                print(f"⚠️  Missing win data for index {img_ind}")
                continue
            
            win_feature = np.array([all_results1[win_path], all_results2[win_path]])
            
            features = np.array(features)
            paths = np.array(paths)
            
            if len(features) == 0:
                print(f"⚠️  No features for index {img_ind}")
                continue
            
            print(f"  Processing index {img_ind} ({len(features)} losing points)...")
            
            # Process each configuration
            for config_name in configs:
                result = process_single_config(
                    config_name,
                    features,
                    paths,
                    win_feature,
                    epsilon,
                    total_k
                )
                
                if result is not None:
                    config_outputs[config_name][str(img_ind)] = result
        
        except Exception as e:
            print(f"❌ Error at index {img_ind}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n💾 Saving outputs...\n")
    
    # Save outputs for each configuration
    for config_name in configs:
        
        if len(config_outputs[config_name]) == 0:
            print(f"⚠️  No data to save for {config_name}")
            continue
        
        output_data = {
            "config": config_name,
            "epsilon": epsilon,
            "total_k": total_k if "diversity" in config_name else None,
            "num_images": len(config_outputs[config_name]),
            "results": config_outputs[config_name]
        }
        
        # Determine if this config uses clustering
        uses_clustering = config_name in ["kmeans_only", "epsilon_kmeans"]
        
        if uses_clustering:
            # Save in a folder structure
            config_dir = os.path.join(output_dir, config_name)
            
            # Create separate files for each cluster
            cluster_outputs = {"cluster_0": {}, "cluster_1": {}, "cluster_2": {}}
            
            for img_ind, clusters in config_outputs[config_name].items():
                if isinstance(clusters, dict):  # Make sure it's a cluster dict
                    for cluster_id in ["cluster_0", "cluster_1", "cluster_2"]:
                        if cluster_id in clusters:
                            cluster_outputs[cluster_id][img_ind] = clusters[cluster_id]
            
            # Save each cluster file
            for cluster_id, data in cluster_outputs.items():
                cluster_file = os.path.join(config_dir, f"{cluster_id}.json")
                with open(cluster_file, "w") as f:
                    json.dump({
                        "config": config_name,
                        "cluster": cluster_id,
                        "epsilon": epsilon,
                        "num_images": len(data),
                        "results": data
                    }, f, indent=4)
                
                print(f"  ✅ {config_name}/{cluster_id}.json ({len(data)} images)")
            
            # Also save a combined file
            combined_file = os.path.join(config_dir, "combined.json")
            with open(combined_file, "w") as f:
                json.dump(output_data, f, indent=4)
            
            print(f"  ✅ {config_name}/combined.json")
        
        else:
            # Save as single file
            output_file = os.path.join(output_dir, f"{config_name}.json")
            with open(output_file, "w") as f:
                json.dump(output_data, f, indent=4)
            
            # Calculate stats for display
            num_imgs = len(config_outputs[config_name])
            total_points = sum(len(v) if isinstance(v, list) else 0 
                             for v in config_outputs[config_name].values())
            avg_points = total_points / num_imgs if num_imgs > 0 else 0
            
            print(f"  ✅ {config_name}.json ({num_imgs} images, avg {avg_points:.1f} points/image)")
    
    print(f"\n🎉 All configurations processed successfully!")
    print(f"📂 Output directory: {output_dir}\n")
    
    # Print summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for config in configs:
        num_imgs = len(config_outputs[config])
        if num_imgs > 0:
            print(f"  {config:30s} : {num_imgs} images processed")
        else:
            print(f"  {config:30s} : No data")
    print("=" * 60)


if __name__ == "__main__":
    
    main(
        json_path1="/workspace/data/Pulkit/pref_qual/asthetic.json",
        json_path2="/workspace/data/Pulkit/pref_qual/imagereward.json",
        base_path="/workspace/data/Pulkit/pref_qual/dataset_full",
        output_dir="/workspace/data/Pulkit/pref_qual/sampling_configs",
        epsilon=0.05,
        total_k=16,
        img_indices=range(400, 405)
    )
```

## **Key Improvements:**

### **1. `create_output_structure()` Function** ✅
- Creates main output directory if it doesn't exist
- Creates subdirectories for clustered configs (`kmeans_only`, `epsilon_kmeans`)
- Runs **before** any processing starts
- Uses `os.makedirs(exist_ok=True)` to avoid errors if directories already exist

### **2. Better Error Handling** ✅
- Checks if output data exists before saving
- Handles empty results gracefully
- Validates that clustered results are actually dictionaries

### **3. Enhanced Logging** 📊
- Progress indicators for each step
- Clear status messages with emojis
- Per-image processing feedback
- Detailed summary at the end

### **4. Metadata Improvements** 📝
Each JSON file now includes:
- `num_images`: Count of processed images
- Statistics about points per image (for non-clustered configs)

### **5. Summary Report** 📈
At the end, prints a table showing:
- Which configs processed data successfully
- How many images each config processed

### **Output Structure Created:**
```
sampling_configs/
├── none.json
├── epsilon_only.json
├── diversity_only.json
├── epsilon_diversity.json
├── kmeans_diversity.json
├── epsilon_kmeans_diversity.json
├── kmeans_only/
│   ├── cluster_0.json
│   ├── cluster_1.json
│   ├── cluster_2.json
│   └── combined.json
└── epsilon_kmeans/
    ├── cluster_0.json
    ├── cluster_1.json
    ├── cluster_2.json
    └── combined.json
```

### **Example Run Output:**
```
📁 Created main output directory: /workspace/data/Pulkit/pref_qual/sampling_configs
📁 Created directory for kmeans_only: /workspace/data/Pulkit/pref_qual/sampling_configs/kmeans_only
📁 Created directory for epsilon_kmeans: /workspace/data/Pulkit/pref_qual/sampling_configs/epsilon_kmeans
✅ Output structure created successfully!

📂 Loading JSON data...
✅ JSON data loaded successfully!

🔄 Processing 5 image indices...
  Processing index 400 (85 losing points)...
  Processing index 401 (87 losing points)...
  ...

💾 Saving outputs...
  ✅ none.json (5 images, avg 85.4 points/image)
  ✅ epsilon_only.json (5 images, avg 62.2 points/image)
  ...
