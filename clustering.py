import os
import sys
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "lib")
)
from data_loading import make_splits
from analysis import make_analysis, plot_cluster_spread

THIS_PATH = os.path.dirname(os.path.realpath(__file__))


def transformer_kmeans_clustering(dataset, k=50):
    texts = [text for text, _ in dataset]
    ratings = np.array([rating for _, rating in dataset])

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(
        texts, show_progress_bar=True, batch_size=64, convert_to_numpy=True
    ).astype(np.float64)

    # Normalize embeddings for spherical K-Means
    embeddings = normalize(embeddings, norm="l2", axis=1)

    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=256)
    clusters = kmeans.fit_predict(embeddings)

    cluster_ratings = {}
    cluster_stats = []
    for i in range(k):
        cluster_stats.append(
            (np.mean(ratings[clusters == i]), np.std(ratings[clusters == i]))
        )
        cluster_ratings[i] = np.bincount(ratings[clusters == i].astype(int)).argmax()
        # cluster_ratings[i] = np.median(ratings[clusters == i])
        # tried mean, median and max, this one seems to be the best
    return model, kmeans, cluster_ratings, cluster_stats


def transformer_clustering_predict(dataset, model, kmeans, ratings, prefix="", x=15):
    texts = [text for text, _ in dataset]
    embeddings = model.encode(
        texts, show_progress_bar=True, batch_size=64, convert_to_numpy=True
    ).astype(np.float64)
    embeddings = normalize(embeddings, norm="l2", axis=1)

    centers = normalize(kmeans.cluster_centers_, norm="l2", axis=1)
    dists = cdist(embeddings, centers, metric="euclidean")
    nearest_indices = np.argsort(dists, axis=1)[:, :x]
    nearest_dists = np.take_along_axis(dists, nearest_indices, axis=1)

    n_clusters = kmeans.n_clusters
    cluster_scores = np.array([ratings[i] for i in range(n_clusters)], dtype=np.float64)

    weights = 1.0 / (nearest_dists + 1e-8)  # larger weight for closer centers
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
        os.path.join(THIS_PATH, "datasets", "Handmade_Products_f.jsonl")
    )

    model, kmeans, cluster_ratings, stats = transformer_kmeans_clustering(
        train_ds,
        k=2000,
    )
    plot_cluster_spread(
        stats, savepath=os.path.join(THIS_PATH, "plots", "Cluster_Distribution.png")
    )
    # transformer_clustering_predict(test_ds, model, kmeans, cluster_ratings)

    # MSE: 1.095 for test with k=20
