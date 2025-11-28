#!/usr/bin/env python3
"""
Comprehensive test script to verify all imports and basic functionality.
"""
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, PROJECT_ROOT)

def test_imports():
    """Test that all required modules can be imported."""
    print("=" * 60)
    print("Testing Imports")
    print("=" * 60)
    
    errors = []
    
    # Core dependencies
    try:
        import numpy as np
        print("✓ numpy")
    except ImportError as e:
        errors.append(f"numpy: {e}")
        print(f"✗ numpy: {e}")
    
    try:
        import scipy
        print("✓ scipy")
    except ImportError as e:
        errors.append(f"scipy: {e}")
        print(f"✗ scipy: {e}")
    
    try:
        import pandas as pd
        print("✓ pandas")
    except ImportError as e:
        errors.append(f"pandas: {e}")
        print(f"✗ pandas: {e}")
    
    try:
        import sklearn
        print("✓ scikit-learn")
    except ImportError as e:
        errors.append(f"scikit-learn: {e}")
        print(f"✗ scikit-learn: {e}")
    
    try:
        import sentence_transformers
        print("✓ sentence-transformers")
    except ImportError as e:
        errors.append(f"sentence-transformers: {e}")
        print(f"✗ sentence-transformers: {e}")
    
    try:
        import transformers
        print("✓ transformers")
    except ImportError as e:
        errors.append(f"transformers: {e}")
        print(f"✗ transformers: {e}")
    
    try:
        import nltk
        print("✓ nltk")
    except ImportError as e:
        errors.append(f"nltk: {e}")
        print(f"✗ nltk: {e}")
    
    try:
        import torch
        print("✓ torch")
    except ImportError as e:
        errors.append(f"torch: {e}")
        print(f"✗ torch: {e}")
    
    try:
        import matplotlib
        print("✓ matplotlib")
    except ImportError as e:
        errors.append(f"matplotlib: {e}")
        print(f"✗ matplotlib: {e}")
    
    try:
        import seaborn
        print("✓ seaborn")
    except ImportError as e:
        errors.append(f"seaborn: {e}")
        print(f"✗ seaborn: {e}")
    
    try:
        import wordcloud
        print("✓ wordcloud")
    except ImportError as e:
        errors.append(f"wordcloud: {e}")
        print(f"✗ wordcloud: {e}")
    
    try:
        import networkx as nx
        print("✓ networkx")
    except ImportError as e:
        errors.append(f"networkx: {e}")
        print(f"✗ networkx: {e}")
    
    try:
        import tqdm
        print("✓ tqdm")
    except ImportError as e:
        errors.append(f"tqdm: {e}")
        print(f"✗ tqdm: {e}")
    
    # Project modules
    print("\n" + "-" * 60)
    print("Testing Project Modules")
    print("-" * 60)
    
    try:
        from src.utils.data_loading import make_splits, ReviewDataset
        print("✓ src.utils.data_loading")
    except ImportError as e:
        errors.append(f"src.utils.data_loading: {e}")
        print(f"✗ src.utils.data_loading: {e}")
    
    try:
        from src.utils.analysis import make_analysis, plot_cluster_spread
        print("✓ src.utils.analysis")
    except ImportError as e:
        errors.append(f"src.utils.analysis: {e}")
        print(f"✗ src.utils.analysis: {e}")
    
    try:
        from src.models.baseline import (
            build_word_stats,
            assign_sentiment,
            plot_word_sentiment_distribution,
            plot_word_cloud,
            evaluate_word_sentiment_reconstruction
        )
        print("✓ src.models.baseline")
    except ImportError as e:
        errors.append(f"src.models.baseline: {e}")
        print(f"✗ src.models.baseline: {e}")
    
    try:
        from src.models.clustering import (
            transformer_kmeans_clustering,
            transformer_clustering_predict
        )
        print("✓ src.models.clustering")
    except ImportError as e:
        errors.append(f"src.models.clustering: {e}")
        print(f"✗ src.models.clustering: {e}")
    
    try:
        from src.models.lsh import MinHash, LSHIndex, get_shingles, estimate_ratings
        print("✓ src.models.lsh")
    except ImportError as e:
        errors.append(f"src.models.lsh: {e}")
        print(f"✗ src.models.lsh: {e}")
    
    try:
        from src.models.tfidf import (
            train_tfidf_model,
            predict_tfidf,
            generate_rating_wordclouds
        )
        print("✓ src.models.tfidf")
    except ImportError as e:
        errors.append(f"src.models.tfidf: {e}")
        print(f"✗ src.models.tfidf: {e}")
    
    try:
        from src.models.transformer import (
            build_transformer,
            encode_texts,
            train_sentence_transformer
        )
        print("✓ src.models.transformer")
    except ImportError as e:
        errors.append(f"src.models.transformer: {e}")
        print(f"✗ src.models.transformer: {e}")
    
    try:
        from src.models.NumpyClusteringImplementation import MyMiniBatchKMeans
        print("✓ src.models.NumpyClusteringImplementation")
    except ImportError as e:
        errors.append(f"src.models.NumpyClusteringImplementation: {e}")
        print(f"✗ src.models.NumpyClusteringImplementation: {e}")
    
    try:
        from src.plotting.common import (
            plot_contrastive_wordclouds,
            plot_five_star_wordclouds,
            plot_review_length_analysis,
            plot_embedding_projection,
            plot_embedding_projection_3D
        )
        print("✓ src.plotting.common")
    except ImportError as e:
        errors.append(f"src.plotting.common: {e}")
        print(f"✗ src.plotting.common: {e}")
    
    return errors


def test_data_files():
    """Test that required data files exist."""
    print("\n" + "=" * 60)
    print("Testing Data Files")
    print("=" * 60)
    
    errors = []
    
    required_files = [
        "datasets/Handmade_Products_f.jsonl",
        "datasets/All_Beauty_f.jsonl",
    ]
    
    for file_path in required_files:
        full_path = os.path.join(PROJECT_ROOT, file_path)
        if os.path.exists(full_path):
            print(f"✓ {file_path}")
        else:
            errors.append(f"Missing: {file_path}")
            print(f"✗ {file_path} (missing)")
    
    # Check optional files
    optional_files = [
        "sentence_tranformer_weights.pth",
        "tokenizer/tokenizer.json",
    ]
    
    for file_path in optional_files:
        full_path = os.path.join(PROJECT_ROOT, file_path)
        if os.path.exists(full_path):
            print(f"✓ {file_path} (optional)")
        else:
            print(f"⚠ {file_path} (optional, missing)")
    
    return errors


def test_basic_functionality():
    """Test basic functionality of key modules."""
    print("\n" + "=" * 60)
    print("Testing Basic Functionality")
    print("=" * 60)
    
    errors = []
    
    # Test data loading
    try:
        from src.utils.data_loading import make_splits
        dataset_path = os.path.join(PROJECT_ROOT, "datasets", "Handmade_Products_f.jsonl")
        if os.path.exists(dataset_path):
            train_ds, val_ds, test_ds = make_splits(dataset_path)
            print(f"✓ Data loading: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
        else:
            print("⚠ Data loading: dataset file not found, skipping test")
    except Exception as e:
        errors.append(f"Data loading: {e}")
        print(f"✗ Data loading: {e}")
    
    # Test baseline model imports
    try:
        from src.models.baseline import build_word_stats, assign_sentiment
        print("✓ Baseline model functions importable")
    except Exception as e:
        errors.append(f"Baseline model: {e}")
        print(f"✗ Baseline model: {e}")
    
    # Test clustering model imports
    try:
        from src.models.clustering import transformer_kmeans_clustering
        print("✓ Clustering model functions importable")
    except Exception as e:
        errors.append(f"Clustering model: {e}")
        print(f"✗ Clustering model: {e}")
    
    # Test LSH model imports
    try:
        from src.models.lsh import MinHash, LSHIndex
        minhash = MinHash(num_hashes=10)
        lsh = LSHIndex(num_hashes=10, bands=5)
        print("✓ LSH model classes instantiable")
    except Exception as e:
        errors.append(f"LSH model: {e}")
        print(f"✗ LSH model: {e}")
    
    # Test TF-IDF model imports
    try:
        from src.models.tfidf import train_tfidf_model
        print("✓ TF-IDF model functions importable")
    except Exception as e:
        errors.append(f"TF-IDF model: {e}")
        print(f"✗ TF-IDF model: {e}")
    
    return errors


def test_script_imports():
    """Test that all scripts can be imported without errors."""
    print("\n" + "=" * 60)
    print("Testing Script Imports")
    print("=" * 60)
    
    errors = []
    
    scripts = [
        "scripts/run_baseline.py",
        "scripts/run_clustering.py",
        "scripts/run_lsh.py",
        "scripts/run_tfidf.py",
        "scripts/generate_plots.py",
    ]
    
    for script_path in scripts:
        full_path = os.path.join(PROJECT_ROOT, script_path)
        if os.path.exists(full_path):
            try:
                # Try to compile the script
                with open(full_path, 'r') as f:
                    code = f.read()
                compile(code, full_path, 'exec')
                print(f"✓ {script_path}")
            except SyntaxError as e:
                errors.append(f"{script_path}: Syntax error - {e}")
                print(f"✗ {script_path}: Syntax error - {e}")
            except Exception as e:
                errors.append(f"{script_path}: {e}")
                print(f"✗ {script_path}: {e}")
        else:
            errors.append(f"{script_path}: File not found")
            print(f"✗ {script_path}: File not found")
    
    return errors


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE REPOSITORY TEST")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}\n")
    
    all_errors = []
    
    # Run all test suites
    all_errors.extend(test_imports())
    all_errors.extend(test_data_files())
    all_errors.extend(test_basic_functionality())
    all_errors.extend(test_script_imports())
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if all_errors:
        print(f"\n✗ Found {len(all_errors)} error(s):")
        for error in all_errors:
            print(f"  - {error}")
        print("\nPlease fix the errors above before running the scripts.")
        return 1
    else:
        print("\n✓ All tests passed! The repository is ready to use.")
        print("\nYou can now run the scripts:")
        print("  - python scripts/run_baseline.py")
        print("  - python scripts/run_clustering.py")
        print("  - python scripts/run_lsh.py")
        print("  - python scripts/run_tfidf.py")
        print("  - python scripts/generate_plots.py")
        return 0


if __name__ == "__main__":
    sys.exit(main())

