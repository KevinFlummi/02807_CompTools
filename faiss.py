import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "lib")
)
from data_loading import make_splits
from analysis import make_analysis
from transformer import build_transformer, encode_texts

THIS_PATH = os.path.dirname(os.path.realpath(__file__))


class IndexFlatIP:
    def __init__(self, dim, normalize_vectors=False):
        self.dim = dim
        self.normalize_vectors = normalize_vectors
        self._embeddings = []
        self._labels = []
        self._matrix = None
        self._built = False

    def _maybe_normalize(self, v):
        v = np.asarray(v, dtype=np.float32)
        if self.normalize_vectors:
            n = np.linalg.norm(v)
            if n > 0:
                v = v / n
        return v

    def _ensure_matrix(self):
        """Stack all embeddings into a search matrix when needed."""
        if not self._built:
            if self._embeddings:
                self._matrix = np.vstack(self._embeddings).astype(np.float32)
            else:
                self._matrix = np.zeros((0, self.dim), dtype=np.float32)
            self._built = True

    def add(self, signature, label):
        """Add one embedding + label. Keeps name 'signature' for compatibility."""
        v = self._maybe_normalize(signature)
        if v.shape[-1] != self.dim:
            raise ValueError(f"Expected dim {self.dim}, got {v.shape}")
        self._embeddings.append(v)
        self._labels.append(float(label))
        self._built = False

    def query(self, signature, top_k=None):
        """Return nearest neighbor indices sorted by similarity."""
        if len(self._embeddings) == 0:
            return []

        self._ensure_matrix()
        q = self._maybe_normalize(signature)
        sims = self._matrix @ q  # dot product

        n = sims.shape[0]
        if top_k is None or top_k >= n:
            idx_sorted = np.argsort(-sims)
        else:
            idx_part = np.argpartition(-sims, top_k - 1)[:top_k]
            idx_sorted = idx_part[np.argsort(-sims[idx_part])]

        return idx_sorted.tolist()

    def predict_label(self, signature, top_k=1):
        """Predict rating as similarity-weighted mean of nearest neighbors."""
        if len(self._embeddings) == 0:
            return None, []

        self._ensure_matrix()
        q = self._maybe_normalize(signature)
        sims = self._matrix @ q
        n = sims.shape[0]
        k = min(top_k, n)

        idx_part = np.argpartition(-sims, k - 1)[:k]
        idx_sorted = idx_part[np.argsort(-sims[idx_part])]

        top_sims = sims[idx_sorted]
        top_labels = np.array([self._labels[i] for i in idx_sorted], dtype=np.float32)

        # Clamp negative similarities (cosine can be negative)
        weights = np.maximum(top_sims, 0.0)
        total = float(weights.sum())

        if total == 0:
            avg_label = float(top_labels.mean())
        else:
            avg_label = float((weights * top_labels).sum() / total)

        neighbors = list(zip(top_sims.tolist(), top_labels.tolist()))
        return avg_label, neighbors


def build_embedding_index(
    train_ds, model, tokenizer, cache_dir, cache_name="faiss_embeddings"
):
    os.makedirs(cache_dir, exist_ok=True)

    cache_vec = os.path.join(cache_dir, f"{cache_name}_vecs.npy")
    cache_rat = os.path.join(cache_dir, f"{cache_name}_ratings.npy")

    embeddings_data = None
    ratings_data = None

    if os.path.exists(cache_vec) and os.path.exists(cache_rat):
        print(f"Found cache {cache_vec}, loading...")
        try:
            embeddings_data = np.load(cache_vec)
            ratings_data = np.load(cache_rat)
            print(f"Loaded {len(embeddings_data)} embeddings.")
        except Exception as e:
            print("Error loading cache:", e)

    if embeddings_data is None:
        print("Encoding sentence-transformer embeddings for training set...")
        texts = [text for text, _ in train_ds]
        ratings = [rating for _, rating in train_ds]

        embeddings = encode_texts(model, tokenizer, texts)
        embeddings = normalize(embeddings, norm="l2", axis=1).astype(np.float32)

        embeddings_data = embeddings
        ratings_data = np.array(ratings, dtype=np.float32)

        print("Saving embedding cache...")
        np.save(cache_vec, embeddings_data)
        np.save(cache_rat, ratings_data)

    dim = embeddings_data.shape[1]
    index = IndexFlatIP(dim=dim, normalize_vectors=False)

    print("Building IndexFlatIP model...")
    for vec, rating in zip(embeddings_data, ratings_data):
        index.add(vec, rating)

    return index


def estimate_ratings_with_embeddings(
    dataset, index, model, tokenizer, prefix="", batch_size=64
):
    true_scores = []
    pred_scores = []
    n_unknown = 0

    texts = [text for text, _ in dataset]
    ratings = [rating for _, rating in dataset]

    for start in tqdm(range(0, len(texts), batch_size), desc="Running test"):
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]
        batch_ratings = ratings[start:end]

        # Encode & normalize
        batch_emb = encode_texts(model, tokenizer, batch_texts)
        batch_emb = normalize(batch_emb, norm="l2", axis=1).astype(np.float32)

        for emb, rating in zip(batch_emb, batch_ratings):
            pred, _ = index.predict_label(emb, top_k=5)
            if pred is None:
                pred = 3.0
                n_unknown += 1

            true_scores.append(rating)
            pred_scores.append(pred)

    make_analysis(true_scores, pred_scores, prefix=prefix, suffix="faiss")
    return n_unknown


if __name__ == "__main__":
    # Load datasets
    train_ds, val_ds, test_ds = make_splits(
        os.path.join(THIS_PATH, "datasets", "Handmade_Products_f.jsonl")
    )

    # Build transformer + tokenizer
    model = build_transformer()
    model.load_state_dict(
        torch.load("sentence_tranformer_weights.pth", map_location="cpu")
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("tokenizer/")

    # Build IndexFlatIP embedding index
    cache_dir = os.path.join(THIS_PATH, "cache")
    index = build_embedding_index(train_ds, model, tokenizer, cache_dir)

    # Evaluate on main test set
    unknown = estimate_ratings_with_embeddings(test_ds, index, model, tokenizer)
    print("Unknown:", unknown, "/", len(test_ds))

    # Evaluate on All_Beauty dataset
    _, _, test_ds_beauty = make_splits(
        os.path.join(THIS_PATH, "datasets", "All_Beauty_f.jsonl")
    )
    unknown = estimate_ratings_with_embeddings(
        test_ds_beauty, index, model, tokenizer, prefix="AllBeauty_"
    )
    print("Unknown:", unknown, "/", len(test_ds_beauty))
