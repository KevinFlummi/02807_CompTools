#!/usr/bin/env python3
"""
Entry point script for running LSH model.
"""
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.models.lsh import MinHash, LSHIndex, estimate_ratings
from src.utils.data_loading import make_splits

if __name__ == "__main__":
    train_ds, val_ds, test_ds = make_splits(
        os.path.join(PROJECT_ROOT, "datasets", "Handmade_Products_f.jsonl")
    )
    
    NUM_HASHES = 180
    BANDS = 60
    
    minhash = MinHash(num_hashes=NUM_HASHES)
    lsh = LSHIndex(num_hashes=NUM_HASHES, bands=BANDS)

    cache_dir = os.path.join(PROJECT_ROOT, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    import numpy as np
    from tqdm import tqdm
    from src.models.lsh import get_shingles
    
    cache_sig_path = os.path.join(cache_dir, "lsh_signatures.npy")
    cache_rat_path = os.path.join(cache_dir, "lsh_ratings.npy")
    
    signatures_data = None
    ratings_data = None
    
    if os.path.exists(cache_sig_path) and os.path.exists(cache_rat_path):
        print(f"Trovata cache! Caricamento firme da {cache_sig_path}...")
        try:
            signatures_data = np.load(cache_sig_path)
            ratings_data = np.load(cache_rat_path)
            print(f"Caricate {len(signatures_data)} firme dalla cache.")
        except Exception as e:
            print(f"Errore caricamento cache: {e}. Ricalcolo...")

    if signatures_data is None:
        print("Calcolo firme LSH in corso (sarà lento la prima volta)...")
        sig_list = []
        rat_list = []
        
        for text, rating in tqdm(train_ds, desc="Hashing"):
            shingles = get_shingles(text)
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

    print("Costruzione indice LSH...")
    num_items = len(signatures_data)
    for i in tqdm(range(num_items), desc="Populating Index"):
        lsh.add(signatures_data[i], ratings_data[i])

    unknown = estimate_ratings(test_ds, lsh, minhash)
    print("Number of unknown scores:", unknown, "out of", len(test_ds))

    _, _, test_ds_beauty = make_splits(
        os.path.join(PROJECT_ROOT, "datasets", "All_Beauty_f.jsonl")
    )
    unknown = estimate_ratings(test_ds_beauty, lsh, minhash, prefix="AllBeauty_")
    print("Number of unknown scores:", unknown, "out of", len(test_ds_beauty))

