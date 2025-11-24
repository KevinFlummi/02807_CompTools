import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from sklearn.decomposition import PCA
from tqdm import tqdm
import re

# Importiamo dai tuoi file esistenti
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from data_loading import make_splits

# Setup percorsi
THIS_PATH = os.path.dirname(os.path.realpath(__file__))
PLOTS_DIR = os.path.join(THIS_PATH, "new_plots_All_Beauty")
os.makedirs(PLOTS_DIR, exist_ok=True)

def clean_text(text):
    """Pulizia veloce per le wordcloud"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def plot_contrastive_wordclouds(dataset, save_name="Contrastive_WordCloud.png"):
    """
    Genera due wordcloud affiancate: 
    Sinistra: Parole frequenti nelle recensioni 1 stella (Negative)
    Destra: Parole frequenti nelle recensioni 5 stelle (Positive)
    """
    print("Generazione Wordclouds Contrastive...")
    # Qui dobbiamo iterare manualmente per evitare errori di slicing su ReviewDataset
    neg_texts = []
    pos_texts = []
    
    # Leggiamo tutto il dataset una volta (potrebbe impiegare un po' ma è sicuro)
    limit = len(dataset)
        
    print(f"Analisi di {limit} recensioni per le wordcloud...")
    for i in range(limit):
        text, rating = dataset[i]
        if rating == 1.0:
            neg_texts.append(clean_text(text))
        elif rating == 5.0:
            pos_texts.append(clean_text(text))

    neg_text_blob = " ".join(neg_texts)
    pos_text_blob = " ".join(pos_texts)

    # Usiamo le stopwords di default + alcune specifiche di Amazon
    stops = set(STOPWORDS)
    stops.update(["product", "item", "one", "will", "bought", "amazon"])

    wc_kwargs = {
        'background_color': 'white',
        'stopwords': stops,
        'max_words': 100,
        'width': 800,
        'height': 600
    }

    # Gestione caso liste vuote
    if not neg_text_blob: neg_text_blob = "no_data"
    if not pos_text_blob: pos_text_blob = "no_data"

    wc_neg = WordCloud(colormap="magma", **wc_kwargs).generate(neg_text_blob)
    wc_pos = WordCloud(colormap="viridis", **wc_kwargs).generate(pos_text_blob)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    axes[0].imshow(wc_neg, interpolation='bilinear')
    axes[0].set_title("1-Star Reviews (Negative Context)", fontsize=16)
    axes[0].axis("off")

    axes[1].imshow(wc_pos, interpolation='bilinear')
    axes[1].set_title("5-Star Reviews (Positive Context)", fontsize=16)
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, save_name), dpi=300)
    print(f"Salvato: {save_name}")

def plot_five_star_wordclouds(dataset, save_name="All_Ratings_WordCloud.png"):
    """
    Genera 5 wordcloud, una per ogni voto da 1 a 5.
    Disposizione a griglia 2x3 (l'ultimo slot rimarrà vuoto).
    """
    print("Generazione Wordclouds per tutti i 5 rating...")
    
    # 1. Raccogliamo i testi divisi per rating
    # Usiamo un dizionario di liste
    texts_by_rating = {1: [], 2: [], 3: [], 4: [], 5: []}
    
    # Limitiamo a 20k per velocità, ma puoi aumentarlo
    limit = len(dataset)
    print(f"Analisi di {limit} recensioni...")
    
    for i in range(limit):
        text, rating = dataset[i]
        r = int(rating)
        if r in texts_by_rating:
            texts_by_rating[r].append(clean_text(text))

    # 2. Setup Colori per emozione
    # 1=Rosso, 2=Arancio, 3=Blu(Neutro), 4=VerdeAcqua, 5=VerdePuro
    colormaps = {1: 'Reds', 2: 'Oranges', 3: 'Blues', 4: 'winter', 5: 'Greens'}
    titles = {
        1: "1 Star: Very Negative",
        2: "2 Stars: Negative",
        3: "3 Stars: Neutral/Mixed",
        4: "4 Stars: Positive",
        5: "5 Stars: Excellent"
    }

    # 3. Creiamo la griglia 2 righe x 3 colonne
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten() # Appiattiamo per accedervi come lista (0,1,2,3,4,5)

    stops = set(STOPWORDS)
    stops.update(["product", "item", "one", "will", "bought", "amazon", "return"])

    wc_kwargs = {
        'background_color': 'white',
        'stopwords': stops,
        'max_words': 80, # Meno parole per renderle più leggibili
        'width': 600,
        'height': 400
    }

    # Ciclo da 1 a 5
    for rating in range(1, 6):
        ax = axes[rating - 1] # 1 stella va all'indice 0
        text_blob = " ".join(texts_by_rating[rating])
        
        if not text_blob:
            text_blob = "No Data"

        # Usiamo il colormap specifico per quel voto
        wc = WordCloud(colormap=colormaps[rating], **wc_kwargs).generate(text_blob)
        
        ax.imshow(wc, interpolation='bilinear')
        ax.set_title(titles[rating], fontsize=14, fontweight='bold')
        ax.axis("off")

    # 4. Nascondiamo il 6° grafico (vuoto)
    axes[5].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, save_name), dpi=300)
    print(f"Salvato: {save_name}")

def plot_review_length_analysis(dataset, save_name="Length_vs_Rating.png"):
    """
    Analizza se c'è una correlazione tra la lunghezza della recensione e il voto.
    """
    print("Analisi Lunghezza vs Rating...")
    ratings = []
    lengths = []

    # Prendiamo un campione di max 10.000 recensioni per fare presto
    limit = len(dataset)
    
    for i in range(limit):
        text, rating = dataset[i]
        ratings.append(int(rating))
        lengths.append(len(text.split()))

    plt.figure(figsize=(10, 6))
    # Fix warning: assegniamo hue=ratings e legend=False
    sns.boxplot(x=ratings, y=lengths, hue=ratings, palette="coolwarm", showfliers=False, legend=False)
    
    plt.title("Review Length Distribution by Rating", fontsize=14)
    plt.xlabel("Star Rating")
    plt.ylabel("Number of Words per Review")
    plt.grid(axis='y', alpha=0.3)
    
    plt.savefig(os.path.join(PLOTS_DIR, save_name), dpi=300)
    print(f"Salvato: {save_name}")

def plot_embedding_projection(dataset, save_name="Embeddings_PCA_2D.png", sample_size=2000):
    """
    Usa gli embeddings già calcolati (cache) o il modello BERT per creare la proiezione 2D.
    """
    print("Generazione Proiezione 2D degli Embeddings...")

    limit = min(len(dataset), sample_size)
    subset_ratings = []
    print(f"Recupero dei primi {limit} voti...")
    for i in range(limit):
        _, rating = dataset[i]
        subset_ratings.append(rating)

    embeddings = None
    cache_path = os.path.join(THIS_PATH, "embeddings_cache.npy")

    # 1. PROVIAMO A USARE LA CACHE (Molto veloce)
    if os.path.exists(cache_path):
        print(f"Trovata cache embeddings! Carico {cache_path}...")
        try:
            full_embeddings = np.load(cache_path)
            # Prendiamo solo le prime 'limit' righe
            embeddings = full_embeddings[:limit]
            print(f"Embeddings caricati dalla cache. Shape: {embeddings.shape}")
        except Exception as e:
            print(f"Errore lettura cache: {e}")

    # 2. SE NON C'È LA CACHE, DOBBIAMO CALCOLARE (Lento e richiede librerie funzionanti)
    if embeddings is None:
        print("Cache non trovata o inutilizzabile. Provo a importare SentenceTransformer...")
        try:
            from sentence_transformers import SentenceTransformer
            # Carichiamo solo i testi necessari
            subset_texts = [dataset[i][0] for i in range(limit)]
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode(subset_texts, show_progress_bar=True, batch_size=64, convert_to_numpy=True)
        except Exception as e:
            print(f"\n[ERRORE FATALE] Non posso generare il grafico.\nManca il file cache E le librerie danno errore: {e}")
            return

    # 3. PCA E PLOT (Uguale a prima)
    try:
        print("Eseguo PCA...")
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embeddings)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            reduced[:, 0], 
            reduced[:, 1], 
            c=subset_ratings, 
            cmap='Spectral', 
            alpha=0.6, 
            s=15,
            edgecolor='none'
        )
        
        cbar = plt.colorbar(scatter, ticks=[1, 2, 3, 4, 5])
        cbar.set_label('Star Rating')
        
        plt.title(f"Semantic Space Projection (PCA) - {limit} samples", fontsize=14)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        
        plt.savefig(os.path.join(PLOTS_DIR, save_name), dpi=300)
        print(f"Salvato: {save_name}")
        
    except Exception as e:
        print(f"Errore durante il plotting: {e}")

def plot_embedding_projection_3D(dataset, save_name="Embeddings_PCA_3D.png", sample_size=2000):
    """
    Usa gli embeddings per creare una proiezione 3D interattiva (salvata come immagine statica).
    """
    print("Generazione Proiezione 3D degli Embeddings...")

    # 1. Recupero Dati
    limit = min(len(dataset), sample_size)
    subset_ratings = []
    print(f"Recupero dei primi {limit} voti...")
    for i in range(limit):
        _, rating = dataset[i]
        subset_ratings.append(rating)

    # 2. Caricamento Embeddings (Cache o Calcolo)
    embeddings = None
    cache_path = os.path.join(THIS_PATH, "embeddings_cache.npy")

    if os.path.exists(cache_path):
        print(f"Trovata cache embeddings! Carico {cache_path}...")
        try:
            full_embeddings = np.load(cache_path)
            embeddings = full_embeddings[:limit]
        except Exception as e:
            print(f"Errore lettura cache: {e}")

    if embeddings is None:
        print("Cache non trovata. Provo a importare SentenceTransformer...")
        try:
            from sentence_transformers import SentenceTransformer
            subset_texts = [dataset[i][0] for i in range(limit)]
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode(subset_texts, show_progress_bar=True, batch_size=64, convert_to_numpy=True)
        except Exception as e:
            print(f"[ERRORE] Impossibile calcolare embeddings: {e}")
            return

    # 3. PCA a 3 Componenti e Plot 3D
    try:
        print("Eseguo PCA a 3 componenti...")
        # --- MODIFICA QUI: n_components=3 ---
        pca = PCA(n_components=3)
        reduced = pca.fit_transform(embeddings)

        # Creazione della figura 3D
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(
            reduced[:, 0], # PC1
            reduced[:, 1], # PC2
            reduced[:, 2], # PC3
            c=subset_ratings, 
            cmap='Spectral', 
            alpha=0.6, 
            s=20,
            edgecolor='none'
        )
        
        cbar = plt.colorbar(scatter, ticks=[1, 2, 3, 4, 5], shrink=0.6, aspect=20)
        cbar.set_label('Star Rating')
        
        ax.set_title(f"3D Semantic Space Projection - {limit} samples", fontsize=14)
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_zlabel("PC 3")
        
        # Angolazione della telecamera (opzionale, per vedere meglio)
        ax.view_init(elev=20, azim=20)

        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, save_name), dpi=300)
        print(f"Salvato: {save_name}")
        
        # Se vuoi vederlo ruotare a schermo, decommenta questa riga:
        # plt.show() 
        
    except Exception as e:
        print(f"Errore durante il plotting 3D: {e}")

if __name__ == "__main__":
    # Carichiamo i dati
    json_path = os.path.join(THIS_PATH, "datasets", "All_Beauty_f.jsonl")
    
    if not os.path.exists(json_path):
        print("Dataset non trovato! Assicurati di aver scaricato i dati.")
        # Dati dummy per testare se non hai il file
        class DummyDS:
            def __len__(self): return 300
            def __getitem__(self, i): return ("Test review text " + str(i), float((i % 5) + 1))
        train_ds = DummyDS()
    else:
        train_ds, _, _ = make_splits(json_path)

    # Eseguiamo le funzioni
    plot_contrastive_wordclouds(train_ds)
    plot_five_star_wordclouds(train_ds)
    plot_review_length_analysis(train_ds)
    
    # Questo proverà a fare il grafico complesso.
    # Se hai ancora problemi di librerie, stamperà l'errore ma NON crasherà.
    plot_embedding_projection(train_ds, sample_size=302977)
    plot_embedding_projection_3D(train_ds, sample_size=302977)

    print("\n=== Fatto! Controlla la cartella 'plots' ===")