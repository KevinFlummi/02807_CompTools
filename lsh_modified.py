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
    
    # Parametri LSH
    NUM_HASHES = 180
    BANDS = 60
    
    minhash = MinHash(num_hashes=NUM_HASHES)
    lsh = LSHIndex(num_hashes=NUM_HASHES, bands=BANDS)

    # --- INIZIO CACHING ---
    cache_sig_path = os.path.join(THIS_PATH, "lsh_signatures.npy")
    cache_rat_path = os.path.join(THIS_PATH, "lsh_ratings.npy")
    
    signatures_data = None
    ratings_data = None
    
    # 1. Controlliamo se esistono i file cache
    if os.path.exists(cache_sig_path) and os.path.exists(cache_rat_path):
        print(f"Trovata cache! Caricamento firme da {cache_sig_path}...")
        try:
            signatures_data = np.load(cache_sig_path)
            ratings_data = np.load(cache_rat_path)
            print(f"Caricate {len(signatures_data)} firme dalla cache.")
        except Exception as e:
            print(f"Errore caricamento cache: {e}. Ricalcolo...")

    # 2. Se non abbiamo caricato (o fallito), calcoliamo da zero
    if signatures_data is None:
        print("Calcolo firme LSH in corso (sarà lento la prima volta)...")
        sig_list = []
        rat_list = []
        
        for text, rating in tqdm(train_ds, desc="Hashing"):
            shingles = get_shingles(text)
            # Gestione casi vuoti (importante per non disallineare gli array)
            if not shingles: 
                continue 
                
            sig = minhash.signature(shingles)
            sig_list.append(sig)
            rat_list.append(rating)
        
        signatures_data = np.array(sig_list)
        ratings_data = np.array(rat_list)
        
        print("Salvataggio cache...")
        np.save(cache_sig_path, signatures_data)
        np.save(cache_rat_path, ratings_data)

    # 3. Popolamento Indice (Veloce)
    # L'indice va ricostruito in RAM ogni volta, ma con i dati pronti è veloce
    print("Costruzione indice LSH...")
    num_items = len(signatures_data)
    for i in tqdm(range(num_items), desc="Populating Index"):
        lsh.add(signatures_data[i], ratings_data[i])
    # --- FINE CACHING ---

    # Ora eseguiamo i test come prima
    unknown = estimate_ratings(test_ds, lsh, minhash)
    print("Number of unknown scores:", unknown, "out of", len(test_ds))

    # Test su All_Beauty (Qui non usiamo cache perché è un test rapido sul test set)
    _, _, test_ds_beauty = make_splits(
        os.path.join(THIS_PATH, "datasets", "All_Beauty_f.jsonl")
    )
    unknown = estimate_ratings(test_ds_beauty, lsh, minhash, prefix="AllBeauty_")
    print("Number of unknown scores:", unknown, "out of", len(test_ds_beauty))