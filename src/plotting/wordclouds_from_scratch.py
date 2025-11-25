import matplotlib.pyplot as plt
import numpy as np
import re
import math
import random
from collections import Counter
import nltk
from nltk.corpus import stopwords
import os
import sys

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# Setup paths for saving plots
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots", "Wordclouds_from_Scratch")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Download stopwords if missing
nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))
# Add custom stopwords specific to Amazon/Reviews
STOPWORDS.update(["product", "item", "one", "will", "bought", "amazon", "review", "use", "return", "really", "get"])

class BoundingBox:
    """Represents the rectangle occupied by a word for collision detection."""
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        
        # Boundary coordinates
        self.left = x - w / 2
        self.right = x + w / 2
        self.top = y + h / 2
        self.bottom = y - h / 2

    def is_overlapping(self, other):
        """AABB Collision check."""
        return not (self.right < other.left or 
                    self.left > other.right or 
                    self.bottom > other.top or 
                    self.top < other.bottom)

def clean_and_count(text, max_words=80):
    """Tokenizes, cleans, and counts words."""
    if not text: return []
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    filtered_words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    counter = Counter(filtered_words)
    return counter.most_common(max_words)

def generate_wordcloud_on_axis(text, ax, colormap_name="viridis", title=""):
    """
    Core Algorithm: Positions words on an Archimedean spiral and draws them on a given Matplotlib Axis.
    """
    print(f"  > Processing spiral for: {title}...")
    
    # 1. Analyze text
    word_counts = clean_and_count(text, max_words=60)
    
    # Handle empty text case
    if not word_counts:
        ax.text(0.5, 0.5, "No Data", ha='center', va='center', fontsize=20, transform=ax.transAxes)
        ax.axis("off")
        ax.set_title(title)
        return

    # 2. Font sizing setup
    max_freq = word_counts[0][1]
    min_freq = word_counts[-1][1]
    
    def get_font_size(freq):
        # Linear scaling: min 8, max 45 (smaller than before to fit subplots)
        min_size, max_size = 8, 45
        if max_freq == min_freq: return max_size
        return min_size + (freq - min_freq) * (max_size - min_size) / (max_freq - min_freq)

    # 3. Positioning Loop
    occupied_boxes = []
    plotted_words = [] # Stores (x, y, word, fontsize, color)

    step_angle = 0.15 # Radians step
    step_radius = 0.8 # Spiral growth rate

    # Get the color map
    cmap = plt.get_cmap(colormap_name)

    for i, (word, freq) in enumerate(word_counts):
        fs = get_font_size(freq)
        
        # Heuristic for text dimensions (crucial for collision)
        w_est = len(word) * fs * 0.5 
        h_est = fs * 1.0 
        
        angle = 0.0
        radius = 0.0
        
        # Spiral Search
        while True:
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            
            new_box = BoundingBox(x, y, w_est, h_est)
            
            collision = False
            for box in occupied_boxes:
                if new_box.is_overlapping(box):
                    collision = True
                    break
            
            if not collision:
                occupied_boxes.append(new_box)
                # Assign color based on rank (more frequent = darker/stronger color usually)
                color_idx = 1.0 - (i / len(word_counts)) * 0.6 # varies from 1.0 to 0.4
                color = cmap(color_idx)
                
                plotted_words.append((x, y, word, fs, color))
                break
            
            angle += step_angle
            radius += step_radius * 0.05
            
            # Safety break
            if radius > 400: break 

    # 4. Drawing on the specific Axis
    for x, y, word, fs, color in plotted_words:
        ax.text(x, y, word, ha='center', va='center', fontsize=fs, color=color, fontweight='bold')

    # 5. Adjust Axis Limits
    if occupied_boxes:
        all_x = [b.x for b in occupied_boxes]
        all_y = [b.y for b in occupied_boxes]
        margin = 10
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight='bold')


# --- HIGH LEVEL PLOTTING FUNCTIONS ---

def plot_contrastive_scratch(dataset, save_name="Scratch_Contrastive.png"):
    """
    Generates 2 WordClouds (1-star vs 5-star) side-by-side using the custom algorithm.
    """
    print(f"\nGenerating Contrastive WordCloud (From Scratch)...")
    
    # Collect text
    neg_texts = []
    pos_texts = []
    
    # Limit for speed
    limit = min(len(dataset), 20000)
    for i in range(limit):
        text, rating = dataset[i]
        if rating == 1.0: neg_texts.append(text)
        elif rating == 5.0: pos_texts.append(text)
            
    # Setup Figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot Negative (Reds)
    generate_wordcloud_on_axis(
        " ".join(neg_texts), 
        axes[0], 
        colormap_name="Reds", 
        title="1-Star Reviews (Negative)"
    )
    
    # Plot Positive (Greens)
    generate_wordcloud_on_axis(
        " ".join(pos_texts), 
        axes[1], 
        colormap_name="Greens", 
        title="5-Star Reviews (Positive)"
    )
    
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, save_name)
    plt.savefig(save_path, dpi=200)
    print(f"Saved: {save_path}")


def plot_all_ratings_scratch(dataset, save_name="Scratch_All_Ratings.png"):
    """
    Generates 5 WordClouds (one per rating) in a grid using the custom algorithm.
    """
    print(f"\nGenerating All Ratings WordClouds (From Scratch)...")
    
    # Collect text buckets
    texts_by_rating = {1: [], 2: [], 3: [], 4: [], 5: []}
    limit = min(len(dataset), 20000)
    
    for i in range(limit):
        text, rating = dataset[i]
        r = int(rating)
        if r in texts_by_rating:
            texts_by_rating[r].append(text)
            
    # Setup Colors
    # Note: 'winter' must be lowercase in matplotlib
    colormaps = {1: 'Reds', 2: 'Oranges', 3: 'Blues', 4: 'winter', 5: 'Greens'}
    titles = {
        1: "1 Star", 2: "2 Stars", 3: "3 Stars", 4: "4 Stars", 5: "5 Stars"
    }

    # Setup Grid (2 rows, 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()
    
    for rating in range(1, 6):
        ax = axes_flat[rating - 1]
        text_blob = " ".join(texts_by_rating[rating])
        
        generate_wordcloud_on_axis(
            text_blob, 
            ax, 
            colormap_name=colormaps[rating], 
            title=titles[rating]
        )
        
    # Hide the 6th empty plot
    axes_flat[5].axis("off")
    
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, save_name)
    plt.savefig(save_path, dpi=200)
    print(f"Saved: {save_path}")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    from src.utils.data_loading import make_splits

    json_path = os.path.join(PROJECT_ROOT, "datasets", "Handmade_Products_f.jsonl")

    if os.path.exists(json_path):
        train_ds, _, _ = make_splits(json_path)
        
        # 1. Run Contrastive (1 vs 5)
        plot_contrastive_scratch(train_ds)
        
        # 2. Run All Ratings (1 to 5)
        plot_all_ratings_scratch(train_ds)
        
    else:
        print("Dataset not found. Please download the jsonl file.")