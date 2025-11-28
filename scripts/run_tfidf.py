#!/usr/bin/env python3
"""
Entry point script for running TF-IDF model.
"""
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.models.tfidf import train_tfidf_model, predict_tfidf, generate_rating_wordclouds
from src.utils.data_loading import make_splits

if __name__ == "__main__":
    train_ds, val_ds, test_ds = make_splits(
        os.path.join(PROJECT_ROOT, "datasets", "Handmade_Products_f.jsonl")
    )
    
    print("=" * 60)
    print("TF-IDF with Ridge Regression")
    print("=" * 60)
    vectorizer_ridge, model_ridge, _ = train_tfidf_model(
        train_ds, method="ridge", max_features=10000, ngram_range=(1, 2)
    )
    predict_tfidf(test_ds, vectorizer_ridge, model_ridge, "ridge")
    
    print("\n" + "=" * 60)
    print("TF-IDF with Logistic Regression")
    print("=" * 60)
    vectorizer_log, model_log, _ = train_tfidf_model(
        train_ds, method="logistic", max_features=10000, ngram_range=(1, 2)
    )
    predict_tfidf(test_ds, vectorizer_log, model_log, "logistic")
    
    print("\n" + "=" * 60)
    print("Testing on AllBeauty dataset (Ridge)")
    print("=" * 60)
    _, _, test_ds_allbeauty = make_splits(
        os.path.join(PROJECT_ROOT, "datasets", "All_Beauty_f.jsonl")
    )
    predict_tfidf(test_ds_allbeauty, vectorizer_ridge, model_ridge, "ridge", prefix="AllBeauty_")
    
    print("\n" + "=" * 60)
    print("Generating Word Clouds by Rating Community")
    print("=" * 60)
    generate_rating_wordclouds(train_ds, prefix="", top_words=100)

