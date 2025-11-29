import os
import sys
import torch
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

from src.models.NumpyClusteringImplementation import (
    MyMiniBatchKMeans as MiniBatchKMeans,
)
from src.utils.data_loading import make_splits
from src.utils.analysis import make_analysis, plot_cluster_spread
from src.models.transformer import build_transformer, encode_texts


def transformer_kmeans_clustering(
    dataset, model, tokenizer, k=50, cache_name="kmeans_embeddings", cache_dir=None
):
    """
    Perform K-means clustering on transformer embeddings.

    Args:
        dataset: Dataset of (text, rating) pairs
        k: Number of clusters
        cache_suffix: Suffix for cache file name
        cache_dir: Directory for cache files (defaults to cache/ in project root)

    Returns:
        model, kmeans, cluster_ratings, cluster_stats
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_vec = os.path.join(cache_dir, f"{cache_name}_vecs.npy")

    embeddings = None
    ratings = np.array([rating for _, rating in dataset])

    if os.path.exists(cache_vec):
        print(f"Found cache {cache_vec}, loading...")
        try:
            embeddings = np.load(cache_vec)
            print(f"Loaded {len(embeddings)} embeddings.")
        except Exception as e:
            print("Error loading cache:", e)

    if embeddings is None:
        print("Encoding sentence-transformer embeddings for dataset...")
        texts = [text for text, _ in dataset]

        embeddings = encode_texts(model, tokenizer, texts)
        embeddings = normalize(embeddings, norm="l2", axis=1)
        print("Saving embedding cache...")
        np.save(cache_vec, embeddings)

    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=3072)
    clusters = kmeans.fit_predict(embeddings)

    cluster_ratings = {}
    cluster_stats = []
    for i in range(k):
        current_cluster_ratings = ratings[clusters == i]

        if len(current_cluster_ratings) == 0:
            cluster_ratings[i] = 3.0
            cluster_stats.append((3.0, 0.0, 3.0))
            continue

        cluster_ratings[i] = np.bincount(current_cluster_ratings.astype(int)).argmax()

        cluster_stats.append(
            (
                np.mean(current_cluster_ratings),
                np.std(current_cluster_ratings),
                cluster_ratings[i],
            )
        )
    return model, kmeans, cluster_ratings, cluster_stats


def transformer_clustering_predict(
    dataset, model, tokenizer, kmeans, ratings, prefix="", x=15
):
    texts = [text for text, _ in dataset]

    embeddings = encode_texts(model, tokenizer, texts)
    embeddings = normalize(embeddings, norm="l2", axis=1)

    centers = normalize(kmeans.cluster_centers_, norm="l2", axis=1)
    dists = cdist(embeddings, centers, metric="euclidean")
    nearest_indices = np.argsort(dists, axis=1)[:, :x]
    nearest_dists = np.take_along_axis(dists, nearest_indices, axis=1)

    n_clusters = kmeans.n_clusters
    cluster_scores = np.array([ratings[i] for i in range(n_clusters)], dtype=np.float64)

    weights = 1.0 / (nearest_dists + 1e-8)
    weights_sum = weights.sum(axis=1, keepdims=True)
    weights_sum[weights_sum == 0] = 1.0
    weights = weights / weights_sum

    nearest_scores = cluster_scores[nearest_indices]
    pred_scores = np.sum(nearest_scores * weights, axis=1)
    true_scores = np.array([rating for _, rating in dataset], dtype=np.float64)

    make_analysis(true_scores, pred_scores, prefix=prefix, suffix="kmeans")
    return 0


if __name__ == "__main__":
    train_ds, val_ds, test_ds = make_splits(
        os.path.join(PROJECT_ROOT, "datasets", "Handmade_Products_f.jsonl")
    )

    model = build_transformer()
    model.load_state_dict(
        torch.load(
            os.path.join(PROJECT_ROOT, "sentence_tranformer_weights.pth"),
            map_location="cpu",
        )
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("tokenizer/")

    model, kmeans, cluster_ratings, stats = transformer_kmeans_clustering(
        train_ds,
        model,
        tokenizer,
        k=50,
        cache_dir=os.path.join(PROJECT_ROOT, "cache"),
    )

    plots_dir = os.path.join(PROJECT_ROOT, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_cluster_spread(
        stats, savepath=os.path.join(plots_dir, "Cluster_Distribution.png")
    )
