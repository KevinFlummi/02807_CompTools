#!/usr/bin/env python3
"""
Entry point script for running clustering model.
"""

import os
import sys
import torch
from transformers import AutoTokenizer

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.models.clustering import (
    transformer_kmeans_clustering,
    transformer_clustering_predict,
)
from src.utils.data_loading import make_splits
from src.utils.analysis import plot_cluster_spread
from src.models.transformer import build_transformer

if __name__ == "__main__":
    train_ds, val_ds, test_ds = make_splits(
        os.path.join(PROJECT_ROOT, "datasets", "Handmade_Products_f.jsonl")
    )
    # Build transformer + tokenizer
    model = build_transformer()
    model.load_state_dict(
        torch.load(
            os.path.join(PROJECT_ROOT, "sentence_tranformer_weights.pth"),
            map_location="cpu",
        )
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(PROJECT_ROOT, "tokenizer/"))

    cache_dir = os.path.join(PROJECT_ROOT, "cache")

    model, kmeans, cluster_ratings, stats = transformer_kmeans_clustering(
        train_ds,
        model,
        tokenizer,
        k=50,
        cache_dir=cache_dir,
    )

    plots_dir = os.path.join(PROJECT_ROOT, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_cluster_spread(
        stats, savepath=os.path.join(plots_dir, "Cluster_Distribution.png")
    )

    transformer_clustering_predict(test_ds, model, tokenizer, kmeans, cluster_ratings)

    # Evaluate on All_Beauty dataset
    _, _, test_ds_beauty = make_splits(
        os.path.join(PROJECT_ROOT, "datasets", "All_Beauty_f.jsonl")
    )
    transformer_clustering_predict(
        test_ds_beauty, model, tokenizer, kmeans, cluster_ratings, prefix="AllBeauty_"
    )
