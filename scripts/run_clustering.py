#!/usr/bin/env python3
"""
Entry point script for running clustering model.
"""
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.models.clustering import transformer_kmeans_clustering, transformer_clustering_predict
from src.utils.data_loading import make_splits
from src.utils.analysis import plot_cluster_spread

if __name__ == "__main__":
    train_ds, val_ds, test_ds = make_splits(
        os.path.join(PROJECT_ROOT, "datasets", "Handmade_Products_f.jsonl")
    )

    model, kmeans, cluster_ratings, stats = transformer_kmeans_clustering(
        train_ds,
        k=50,
    )
    
    plots_dir = os.path.join(PROJECT_ROOT, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_cluster_spread(
        stats, savepath=os.path.join(plots_dir, "Cluster_Distribution.png")
    )
    
    transformer_clustering_predict(test_ds, model, kmeans, cluster_ratings)

