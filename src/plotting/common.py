"""
Common plotting functions shared across different plotting scripts.
"""
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from sklearn.decomposition import PCA


def clean_text(text):
    """Pulizia veloce per le wordcloud"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text


def plot_contrastive_wordclouds(dataset, plots_dir, save_name="Contrastive_WordCloud.png"):
    """
    Genera due wordcloud affiancate: 
    Sinistra: Parole frequenti nelle recensioni 1 stella (Negative)
    Destra: Parole frequenti nelle recensioni 5 stelle (Positive)
    """
    print("Generazione Wordclouds Contrastive...")
    neg_texts = []
    pos_texts = []
    
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

    stops = set(STOPWORDS)
    stops.update(["product", "item", "one", "will", "bought", "amazon"])

    wc_kwargs = {
        'background_color': 'white',
        'stopwords': stops,
        'max_words': 100,
        'width': 800,
        'height': 600
    }

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
    plt.savefig(os.path.join(plots_dir, save_name), dpi=300)
    print(f"Salvato: {save_name}")


def plot_five_star_wordclouds(dataset, plots_dir, save_name="All_Ratings_WordCloud.png"):
    """
    Genera 5 wordcloud, una per ogni voto da 1 a 5.
    Disposizione a griglia 2x3 (l'ultimo slot rimarrà vuoto).
    """
    print("Generazione Wordclouds per tutti i 5 rating...")
    
    texts_by_rating = {1: [], 2: [], 3: [], 4: [], 5: []}
    
    limit = len(dataset)
    print(f"Analisi di {limit} recensioni...")
    
    for i in range(limit):
        text, rating = dataset[i]
        r = int(rating)
        if r in texts_by_rating:
            texts_by_rating[r].append(clean_text(text))

    colormaps = {1: 'Reds', 2: 'Oranges', 3: 'Blues', 4: 'winter', 5: 'Greens'}
    titles = {
        1: "1 Star: Very Negative",
        2: "2 Stars: Negative",
        3: "3 Stars: Neutral/Mixed",
        4: "4 Stars: Positive",
        5: "5 Stars: Excellent"
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    stops = set(STOPWORDS)
    stops.update(["product", "item", "one", "will", "bought", "amazon", "return"])

    wc_kwargs = {
        'background_color': 'white',
        'stopwords': stops,
        'max_words': 80,
        'width': 600,
        'height': 400
    }

    for rating in range(1, 6):
        ax = axes[rating - 1]
        text_blob = " ".join(texts_by_rating[rating])
        
        if not text_blob:
            text_blob = "No Data"

        wc = WordCloud(colormap=colormaps[rating], **wc_kwargs).generate(text_blob)
        
        ax.imshow(wc, interpolation='bilinear')
        ax.set_title(titles[rating], fontsize=14, fontweight='bold')
        ax.axis("off")

    axes[5].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, save_name), dpi=300)
    print(f"Salvato: {save_name}")


def plot_review_length_analysis(dataset, plots_dir, save_name="Length_vs_Rating.png"):
    """
    Analizza se c'è una correlazione tra la lunghezza della recensione e il voto.
    """
    print("Analisi Lunghezza vs Rating...")
    ratings = []
    lengths = []

    limit = len(dataset)
    
    for i in range(limit):
        text, rating = dataset[i]
        ratings.append(int(rating))
        lengths.append(len(text.split()))

    plt.figure(figsize=(10, 6))
    sns.boxplot(x=ratings, y=lengths, hue=ratings, palette="coolwarm", showfliers=False, legend=False)
    
    plt.title("Review Length Distribution by Rating", fontsize=14)
    plt.xlabel("Star Rating")
    plt.ylabel("Number of Words per Review")
    plt.grid(axis='y', alpha=0.3)
    
    plt.savefig(os.path.join(plots_dir, save_name), dpi=300)
    print(f"Salvato: {save_name}")


def plot_embedding_projection(dataset, plots_dir, cache_path, save_name="Embeddings_PCA_2D.png", sample_size=2000):
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

    if os.path.exists(cache_path):
        print(f"Trovata cache embeddings! Carico {cache_path}...")
        try:
            full_embeddings = np.load(cache_path)
            embeddings = full_embeddings[:limit]
            print(f"Embeddings caricati dalla cache. Shape: {embeddings.shape}")
        except Exception as e:
            print(f"Errore lettura cache: {e}")

    if embeddings is None:
        print("Cache non trovata o inutilizzabile. Provo a importare SentenceTransformer...")
        try:
            from sentence_transformers import SentenceTransformer
            subset_texts = [dataset[i][0] for i in range(limit)]
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode(subset_texts, show_progress_bar=True, batch_size=64, convert_to_numpy=True)
        except Exception as e:
            print(f"\n[ERRORE FATALE] Non posso generare il grafico.\nManca il file cache E le librerie danno errore: {e}")
            return

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
        
        plt.savefig(os.path.join(plots_dir, save_name), dpi=300)
        print(f"Salvato: {save_name}")
        
    except Exception as e:
        print(f"Errore durante il plotting: {e}")


def plot_embedding_projection_3D(dataset, plots_dir, cache_path, save_name="Embeddings_PCA_3D.png", sample_size=2000):
    """
    Usa gli embeddings per creare una proiezione 3D interattiva (salvata come immagine statica).
    """
    print("Generazione Proiezione 3D degli Embeddings...")

    limit = min(len(dataset), sample_size)
    subset_ratings = []
    print(f"Recupero dei primi {limit} voti...")
    for i in range(limit):
        _, rating = dataset[i]
        subset_ratings.append(rating)

    embeddings = None

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

    try:
        print("Eseguo PCA a 3 componenti...")
        pca = PCA(n_components=3)
        reduced = pca.fit_transform(embeddings)

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(
            reduced[:, 0],
            reduced[:, 1],
            reduced[:, 2],
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
        
        ax.view_init(elev=20, azim=20)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, save_name), dpi=300)
        print(f"Salvato: {save_name}")
        
    except Exception as e:
        print(f"Errore durante il plotting 3D: {e}")

