import os
import sys
import numpy as np
from scipy.spatial.distance import cdist


#### from sklearn.cluster import MiniBatchKMeans
from NumpyClusteringImplementation import MyMiniBatchKMeans as MiniBatchKMeans



from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "lib")
)
from data_loading import make_splits
from analysis import make_analysis, plot_cluster_spread

THIS_PATH = os.path.dirname(os.path.realpath(__file__))


def transformer_kmeans_clustering(dataset, k=50, cache_suffix=""):
    texts = [text for text, _ in dataset]
    ratings = np.array([rating for _, rating in dataset])

    # Percorso dove salvare il file "cache"
    cache_path = os.path.join(
        THIS_PATH,
        f"embeddings_cache_{cache_suffix}.npy"
    )

    # Se il file esiste già, CARICALO (ci mette 1 secondo)
    if os.path.exists(cache_path):
        print(f"Caricamento embeddings da cache: {cache_path} ...")
        embeddings = np.load(cache_path)
        # Carichiamo il modello solo perché serve ritornarlo, ma non calcoliamo nulla
        model = SentenceTransformer("all-MiniLM-L6-v2") 
    
    # Se il file NON esiste, CALCOLALO (ci mette 30 min) e poi SALVALO
    else:
        print("Calcolo embeddings in corso (richiederà tempo)...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(
            texts, show_progress_bar=True, batch_size=64, convert_to_numpy=True
        ).astype(np.float64)
        
        print(f"Salvataggio embeddings in cache: {cache_path}")
        np.save(cache_path, embeddings)

    # Normalize embeddings for spherical K-Means
    embeddings = normalize(embeddings, norm="l2", axis=1)

    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=3072) # batch_size=256
    clusters = kmeans.fit_predict(embeddings)

    cluster_ratings = {}
    cluster_stats = []
    for i in range(k):
        # Prendiamo i rating che appartengono al cluster i
        current_cluster_ratings = ratings[clusters == i]
        
        # Se il cluster è vuoto (nessuna recensione assegnata), mettiamo valori dummy
        if len(current_cluster_ratings) == 0:
            cluster_ratings[i] = 3.0 # Valore neutro di default
            cluster_stats.append((3.0, 0.0, 3.0)) 
            continue # Passa al prossimo cluster

        # Se il cluster NON è vuoto, procediamo col calcolo originale
        cluster_ratings[i] = np.bincount(current_cluster_ratings.astype(int)).argmax()
        
        cluster_stats.append(
            (
                np.mean(current_cluster_ratings),
                np.std(current_cluster_ratings),
                cluster_ratings[i],
            )
        )
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
        k=50, # k=2000
    )
    plot_cluster_spread(
        stats, savepath=os.path.join(THIS_PATH, "plots", "Cluster_Distribution.png")
    )
    # transformer_clustering_predict(test_ds, model, kmeans, cluster_ratings)

    # MSE: 1.095 for test with k=20
