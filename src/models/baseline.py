import os
import sys
import re
import nltk
import math
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import defaultdict
from nltk.corpus import stopwords
from tqdm import tqdm

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.data_loading import make_splits
from src.utils.analysis import make_analysis

# Download stopwords to ./nltk_data
nltk.download("stopwords", quiet=True)
# Load stopwords
stop_words = set(stopwords.words("english"))


def build_word_stats(dataset):
    word_stats = defaultdict(lambda: {"score": 0.0, "count": 0})
    word_pattern = re.compile(r"\b\w+\b")
    for text, rating in tqdm(dataset, desc="Processing reviews"):
        words = [x for x in word_pattern.findall(text.lower()) if x not in stop_words]
        for word in words:
            if len(word) > 1 and word.isalpha():
                word_stats[word]["score"] += rating
                word_stats[word]["count"] += 1

    for k, v in word_stats.items():
        word_stats[k]["avg"] = v["score"] / v["count"]

    return word_stats,


def assign_sentiment(word_stats, dataset):
    """
    Compute sentiment scores for each word using the global mean of ratings
    and optional exponential scaling.
    """
    total_count = 0
    total_score_sum = 0.0
    sum_squared_diff = 0.0

    for _, rating in tqdm(dataset, desc="Calculating global mean and std"):
        total_score_sum += rating
        total_count += 1

    global_mean = total_score_sum / total_count

    for _, rating in tqdm(dataset, desc="Calculating variance"):
        sum_squared_diff += (rating - global_mean) ** 2

    global_std = math.sqrt(sum_squared_diff / total_count)

    for stats in word_stats.values():
        diff = stats["avg"] - global_mean
        stats["sentiment"] = np.sign(diff) * diff**2

    return word_stats, global_mean, global_std


def plot_word_sentiment_distribution(word_stats, bins=50, savefig=True, prefix=""):
    sentiments = [v["sentiment"] for v in word_stats.values()]

    plt.figure(figsize=(8, 5))
    plt.hist(sentiments, bins=bins, color="skyblue", edgecolor="black")
    plt.title("Word Sentiment Distribution")
    plt.xlabel("Sentiment Score")
    plt.ylabel("Number of Words")
    plt.grid(axis="y", alpha=0.75)
    
    plots_dir = os.path.join(PROJECT_ROOT, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    if savefig:
        plt.savefig(
            os.path.join(plots_dir, f"{prefix}_SentimentDistribution_baseline.png")
        )
    else:
        plt.show()


def plot_word_cloud(word_stats, top_words=100, savefig=True, prefix=""):
    top_items = sorted(word_stats.items(), key=lambda x: x[1]["count"], reverse=True)[
        :top_words
    ]
    freq_dict = {k: v["count"] for k, v in top_items}

    min_score = 1.0
    max_score = 5.0
    word_avg_scores = {
        k: min(max(v["avg"], min_score), max_score) for k, v in top_items
    }

    def color_func(word, **kwargs):
        score = word_avg_scores.get(word, 3.0)
        norm_score = (score - min_score) / (max_score - min_score)
        r = int(255 * (1 - norm_score))
        g = int(255 * norm_score)
        b = 0
        return f"rgb({r},{g},{b})"

    wordcloud = WordCloud(
        width=800, height=400, background_color="white"
    ).generate_from_frequencies(freq_dict)

    plt.figure(figsize=(12, 6))
    plt.imshow(
        wordcloud.recolor(color_func=color_func, random_state=42),
        interpolation="bilinear",
    )
    plt.axis("off")
    plt.title(f"Word Cloud of Top {top_words} Words (colored by avg score)")
    
    plots_dir = os.path.join(PROJECT_ROOT, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    if savefig:
        plt.savefig(os.path.join(plots_dir, f"{prefix}_Wordcloud_baseline.png"))
    else:
        plt.show()


def evaluate_word_sentiment_reconstruction(
    dataset, word_stats, global_mean, global_std, prefix=""
):
    true_scores = []
    pred_scores = []
    n_unknown = 0

    for text, rating in tqdm(dataset, desc="Running test"):
        words = [w for w in re.findall(r"\b\w+\b", text.lower()) if w not in stop_words]
        word_sents = [
            word_stats[w]["sentiment"]
            for w in words
            if w in word_stats and abs(word_stats[w]["sentiment"]) > 0.3
        ]

        if word_sents:
            avg_sent = np.mean(word_sents)
            pred = global_mean + avg_sent * global_std
        else:
            pred = global_mean
            n_unknown += len([w for w in words if w not in word_stats])

        pred = max(1.0, min(5.0, pred))

        true_scores.append(rating)
        pred_scores.append(pred)

    make_analysis(true_scores, pred_scores, prefix=prefix, suffix="baseline")

    return n_unknown


if __name__ == "__main__":
    train_ds, _, test_ds = make_splits(
        os.path.join(PROJECT_ROOT, "datasets", "Handmade_Products_f.jsonl")
    )
    total_count = len(train_ds)
    word_stats, global_mean, global_std = assign_sentiment(
        *build_word_stats(train_ds), train_ds
    )

    print("Average review score:", global_mean)
    print("Unique words:", total_count)
    print("Example word stats:", list(word_stats.items())[:5])

    plot_word_sentiment_distribution(word_stats, bins=50)
    plot_word_cloud(word_stats, top_words=100)

    unknown = evaluate_word_sentiment_reconstruction(
        test_ds, word_stats, global_mean, global_std
    )
    print("Number of unknown words:", unknown)

    _, _, test_ds = make_splits(
        os.path.join(PROJECT_ROOT, "datasets", "All_Beauty_f.jsonl")
    )
    unknown = evaluate_word_sentiment_reconstruction(
        test_ds, word_stats, global_mean, global_std, prefix="AllBeauty_"
    )
    print("Number of unknown words:", unknown)

