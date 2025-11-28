import os
import sys

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.data_loading import make_splits
from src.plotting.common import (
    plot_contrastive_wordclouds,
    plot_five_star_wordclouds,
    plot_review_length_analysis,
    plot_embedding_projection,
    plot_embedding_projection_3D
)

# Setup paths
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

if __name__ == "__main__":
    # Carichiamo i dati
    json_path = os.path.join(PROJECT_ROOT, "datasets", "Handmade_Products_f.jsonl")
    
    if not os.path.exists(json_path):
        print("Dataset non trovato! Assicurati di aver scaricato i dati.")
        class DummyDS:
            def __len__(self): return 300
            def __getitem__(self, i): return ("Test review text " + str(i), float((i % 5) + 1))
        train_ds = DummyDS()
    else:
        train_ds, _, _ = make_splits(json_path)

    # Cache path for embeddings
    cache_path = os.path.join(PROJECT_ROOT, "cache", "embeddings_cache.npy")
    
    # Eseguiamo le funzioni
    plot_contrastive_wordclouds(train_ds, PLOTS_DIR)
    plot_five_star_wordclouds(train_ds, PLOTS_DIR)
    plot_review_length_analysis(train_ds, PLOTS_DIR)
    
    # This will try to make the complex graph.
    # If you still have library problems, it will print the error but NOT crash.
    plot_embedding_projection(train_ds, PLOTS_DIR, cache_path, sample_size=149851)
    plot_embedding_projection_3D(train_ds, PLOTS_DIR, cache_path, sample_size=149851)

    print("\n=== Fatto! Controlla la cartella 'plots' ===")

