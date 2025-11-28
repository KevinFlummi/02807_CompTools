#!/usr/bin/env python3
"""
Entry point script for generating wordclouds from scratch (custom algorithm).
"""
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.plotting.wordclouds_from_scratch import (
    plot_contrastive_scratch,
    plot_all_ratings_scratch
)
from src.utils.data_loading import make_splits

if __name__ == "__main__":
    json_path = os.path.join(PROJECT_ROOT, "datasets", "Handmade_Products_f.jsonl")

    if os.path.exists(json_path):
        train_ds, _, _ = make_splits(json_path)
        
        # 1. Run Contrastive (1 vs 5)
        plot_contrastive_scratch(train_ds)
        
        # 2. Run All Ratings (1 to 5)
        plot_all_ratings_scratch(train_ds)
        
        print("\n=== Done! Check plots/Wordclouds_from_Scratch/ ===")
    else:
        print("Dataset not found. Please download the jsonl file.")

