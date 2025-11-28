#!/usr/bin/env python3
"""
Entry point script for running baseline model.
"""
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.models.baseline import (
    build_word_stats,
    assign_sentiment,
    plot_word_sentiment_distribution,
    plot_word_cloud,
    evaluate_word_sentiment_reconstruction
)
from src.utils.data_loading import make_splits

if __name__ == "__main__":
    train_ds, _, test_ds = make_splits(
        os.path.join(PROJECT_ROOT, "datasets", "Handmade_Products_f.jsonl")
    )
    total_count = len(train_ds)
    word_stats, global_mean, global_std = assign_sentiment(
        *build_word_stats(train_ds), train_ds
    )

    print("Average review score:", global_mean)
    print("Unique words:", total_count)
    print("Example word stats:", list(word_stats.items())[:5])

    plot_word_sentiment_distribution(word_stats, bins=50)
    plot_word_cloud(word_stats, top_words=100)

    unknown = evaluate_word_sentiment_reconstruction(
        test_ds, word_stats, global_mean, global_std
    )
    print("Number of unknown words:", unknown)

    _, _, test_ds = make_splits(
        os.path.join(PROJECT_ROOT, "datasets", "All_Beauty_f.jsonl")
    )
    unknown = evaluate_word_sentiment_reconstruction(
        test_ds, word_stats, global_mean, global_std, prefix="AllBeauty_"
    )
    print("Number of unknown words:", unknown)

