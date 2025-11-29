#!/usr/bin/env python3
"""
Entry point script for running FAISS IndexFlatIP model.
"""

import os
import sys
import torch
from transformers import AutoTokenizer

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.models.faiss import build_embedding_index, estimate_ratings_with_embeddings
from src.utils.data_loading import make_splits
from src.models.transformer import build_transformer


if __name__ == "__main__":
    # Load datasets
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

    # Build IndexFlatIP embedding index
    cache_dir = os.path.join(PROJECT_ROOT, "cache")
    index = build_embedding_index(train_ds, model, tokenizer, cache_dir)

    # Evaluate on main test set
    unknown = estimate_ratings_with_embeddings(test_ds, index, model, tokenizer)
    print("Unknown:", unknown, "/", len(test_ds))

    # Evaluate on All_Beauty dataset
    _, _, test_ds_beauty = make_splits(
        os.path.join(PROJECT_ROOT, "datasets", "All_Beauty_f.jsonl")
    )
    unknown = estimate_ratings_with_embeddings(
        test_ds_beauty, index, model, tokenizer, prefix="AllBeauty_"
    )
    print("Unknown:", unknown, "/", len(test_ds_beauty))
