import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm

# Importiamo le tue classi
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from data_loading import make_splits
from lsh import MinHash, LSHIndex, get_shingles

THIS_PATH = os.path.dirname(os.path.realpath(__file__))
PLOTS_DIR = os.path.join(THIS_PATH, "plots_lsh")
os.makedirs(PLOTS_DIR, exist_ok=True)

def analyze_lsh_performance_full(dataset):
    """
    Versione FULL: Calcola similarità su tutto, ma restituisce un grafo gestibile.
    """
    total = len(dataset)
    print(f"Costruzione LSH su TUTTE le {total} recensioni...")
    
    num_hashes = 100
    bands = 20
    minhash = MinHash(num_hashes=num_hashes, seed=42)
    lsh = LSHIndex(num_hashes=num_hashes, bands=bands)
    
    # --- FASE 1: Hashing ---
    print("Fase 1/2: Hashing e Indicizzazione...")
    for i in tqdm(range(total), desc="Indicizzazione"):
        text, rating = dataset[i]
        shingles = get_shingles(text)
        if not shingles: continue 
        sig = minhash.signature(shingles)
        lsh.add(sig, i)

    # --- FASE 2: Ricerca ---
    print("Fase 2/2: Analisi Similarità...")
    similarities = []
    G = nx.Graph()
    
    num_indexed = len(lsh.signatures)
    
    for idx in tqdm(range(num_indexed), desc="Ricerca duplicati"):
        current_sig = lsh.signatures[idx]
        candidates = lsh.query(current_sig)
        
        if len(candidates) > 1:
            original_id_A = lsh.labels[idx]
            G.add_node(original_id_A)
            
            for other_idx in candidates:
                if other_idx <= idx: continue
                
                other_sig = lsh.signatures[other_idx]
                sim = np.mean(current_sig == other_sig)
                
                if sim > 0.6: 
                    original_id_B = lsh.labels[other_idx]
                    similarities.append(sim)
                    G.add_edge(original_id_A, original_id_B, weight=sim)

    print(f"Analisi completata. Trovate {len(similarities)} connessioni rilevanti.")
    return similarities, G

def plot_jaccard_histogram(similarities):
    if not similarities:
        print("Nessuna similarità trovata per l'istogramma.")
        return

    plt.figure(figsize=(10, 6))
    # Scala logaritmica per vedere bene sia i pochi duplicati esatti che i tanti simili
    plt.hist(similarities, bins=30, range=(0.6, 1.0), color='purple', edgecolor='black', alpha=0.7)
    plt.title("Distribution of Jaccard Similarity (Full Dataset)", fontsize=14)
    plt.xlabel("Estimated Jaccard Similarity")
    plt.ylabel("Number of Pairs (Log Scale)")
    plt.yscale('log') 
    plt.grid(axis='y', alpha=0.3, which="both")
    
    savepath = os.path.join(PLOTS_DIR, "LSH_Similarity_Distribution_Full.png")
    plt.savefig(savepath, dpi=300)
    print(f"Salvato: {savepath}")

def plot_similarity_network(G, max_nodes=1000):
    """
    Disegna il grafo. Se ci sono troppi nodi, ne prende solo un campione (max_nodes)
    per evitare che il grafico diventi illeggibile.
    """
    # 1. Rimuoviamo nodi isolati (che non hanno simili)
    G_clean = G.copy()
    nodes_to_remove = [n for n, d in dict(G.degree()).items() if d == 0]
    G_clean.remove_nodes_from(nodes_to_remove)
    
    total_connected_nodes = len(G_clean.nodes)
    if total_connected_nodes == 0:
        print("Grafo vuoto (nessun duplicato trovato), niente da disegnare.")
        return

    # 2. FILTRO: Se ci sono troppi nodi, prendiamo solo i primi 'max_nodes'
    if total_connected_nodes > max_nodes:
        print(f"Grafo troppo grande ({total_connected_nodes} nodi). Mostro solo i primi {max_nodes} per leggibilità...")
        subset_nodes = list(G_clean.nodes)[:max_nodes]
        G_clean = G_clean.subgraph(subset_nodes)

    print(f"Disegno grafo con {len(G_clean.nodes)} nodi...")
    
    plt.figure(figsize=(12, 12))
    
    # Layout "Spring" è il migliore per vedere i cluster
    pos = nx.spring_layout(G_clean, k=0.15, iterations=20, seed=42)
    
    # Disegno
    nx.draw_networkx_nodes(G_clean, pos, node_size=30, node_color="skyblue", edgecolors='blue', alpha=0.7)
    nx.draw_networkx_edges(G_clean, pos, alpha=0.3, edge_color="gray", width=0.8)
    
    plt.title(f"Network of Near-Duplicate Reviews (Subset of {len(G_clean.nodes)})", fontsize=16)
    plt.axis("off")
    
    savepath = os.path.join(PLOTS_DIR, "LSH_Network_Graph_Sample.png")
    plt.savefig(savepath, dpi=300)
    print(f"Salvato: {savepath}")

if __name__ == "__main__":
    # Carica dataset
    json_path = os.path.join(THIS_PATH, "datasets", "Handmade_Products_f.jsonl")
    
    if not os.path.exists(json_path):
        print("Dataset non trovato.")
    else:
        train_ds, _, _ = make_splits(json_path)
        
        # 1. Calcoliamo su TUTTO (per statistiche corrette)
        sims, graph = analyze_lsh_performance_full(train_ds)
        
        # 2. Plot Istogramma (su TUTTI i dati trovati)
        plot_jaccard_histogram(sims)
        
        # 3. Plot Grafo (Limitato a max 1000 nodi per estetica)
        try:
            plot_similarity_network(graph, max_nodes=1000)
        except Exception as e:
            print(f"Errore plotting grafo: {e}")