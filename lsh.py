import os
import re
import sys
import nltk
import hashlib
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from nltk.corpus import stopwords

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "lib")
)
from data_loading import make_splits
from analysis import make_analysis

THIS_PATH = os.path.dirname(os.path.realpath(__file__))

# Download stopwords to ./nltk_data
nltk.download("stopwords")
# Load stopwords
stop_words = set(stopwords.words("english"))


def get_shingles(text, q=3):
    word_pattern = re.compile(r"\b\w+\b")
    text = "_".join(
        [x for x in word_pattern.findall(text.lower()) if x not in stop_words]
    )
    return {text[i : i + q] for i in range(max(0, len(text) - q + 1))}


class MinHash:
    def __init__(self, num_hashes=100, seed=42):
        self.num_hashes = num_hashes
        self.seed = seed
        self.large_prime = (1 << 61) - 1
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, self.large_prime, size=num_hashes, dtype=np.uint64)
        self.b = rng.integers(0, self.large_prime, size=num_hashes, dtype=np.uint64)

    def signature(self, shingles):
        if not shingles:
            return np.zeros(self.num_hashes, dtype=np.uint64)
        shingle_hashes = np.array(
            [
                int(hashlib.md5(s.encode()).hexdigest(), 16) & ((1 << 64) - 1)
                for s in shingles
            ],
            dtype=np.uint64,
        )
        hashes = (np.outer(self.a, shingle_hashes) + self.b[:, None]) % self.large_prime
        return hashes.min(axis=1)


class LSHIndex:
    def __init__(self, num_hashes=100, bands=20):
        self.num_hashes = num_hashes
        self.bands = bands
        self.rows = num_hashes // bands
        self.tables = [defaultdict(list) for _ in range(bands)]
        self.signatures = []
        self.labels = []

    def add(self, signature, label):
        idx = len(self.signatures)
        self.signatures.append(signature)
        self.labels.append(label)
        for b in range(self.bands):
            start = b * self.rows
            end = (b + 1) * self.rows
            band_hash = hashlib.md5(signature[start:end].tobytes()).hexdigest()
            self.tables[b][band_hash].append(idx)

    def query(self, signature):
        candidates = set()
        for b in range(self.bands):
            start = b * self.rows
            end = (b + 1) * self.rows
            band_hash = hashlib.md5(signature[start:end].tobytes()).hexdigest()
            candidates.update(self.tables[b].get(band_hash, []))
        return list(candidates)

    def predict_label(self, signature, top_k=1):
        candidates = self.query(signature)
        if not candidates:
            return None, []

        sims = []
        for idx in candidates:
            # Approx Jaccard similarity via MinHash
            sim = np.mean(self.signatures[idx] == signature)
            sims.append((sim, self.labels[idx]))

        sims.sort(reverse=True)
        top_sims = sims[:top_k]

        # Weighted average of top k labels
        total_sim = sum(sim for sim, _ in top_sims)
        if total_sim == 0:
            avg_label = np.mean([label for _, label in top_sims])
        else:
            avg_label = sum(sim * label for sim, label in top_sims) / total_sim

        return avg_label, top_sims  # return also similarity info


def estimate_ratings(dataset, lsh, minhash, prefix=""):
    true_scores = []
    pred_scores = []
    n_unknown = 0

    for text, rating in tqdm(dataset, desc="Running test"):
        sig = minhash.signature(get_shingles(text))
        pred, _ = lsh.predict_label(sig, 5)
        if pred is None:
            pred = 3.0
            n_unknown += 1
        true_scores.append(rating)
        pred_scores.append(pred)

    make_analysis(true_scores, pred_scores, prefix=prefix, suffix="lsh")

    return n_unknown


if __name__ == "__main__":
    train_ds, val_ds, test_ds = make_splits(
        os.path.join(THIS_PATH, "datasets", "Handmade_Products_f.jsonl")
    )
    minhash = MinHash(num_hashes=180)
    lsh = LSHIndex(num_hashes=180, bands=60)
    for text, rating in tqdm(train_ds, desc="Building LSH"):
        shingles = get_shingles(text)
        sig = minhash.signature(shingles)
        lsh.add(sig, rating)

    unknown = estimate_ratings(test_ds, lsh, minhash)
    print("Number of unknown scores:", unknown, "out of", len(test_ds))

    # Also test the word stat map on a fresh dataset to see if it is (somewhat) universal
    _, _, test_ds = make_splits(
        os.path.join(THIS_PATH, "datasets", "All_Beauty_f.jsonl")
    )
    unknown = estimate_ratings(test_ds, lsh, minhash, prefix="AllBeauty_")
    print("Number of unknown scores:", unknown, "out of", len(test_ds))

    # MSE: 1.130 for test
