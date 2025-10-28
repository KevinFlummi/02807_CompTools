import os
import sys
import re
import nltk
import math
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import defaultdict
from nltk.corpus import stopwords
from tqdm import tqdm

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "lib")
)
from data_loading import make_splits
from analysis import mse_plot, confusion_plot

THIS_PATH = os.path.dirname(os.path.realpath(__file__))


# Download stopwords to ./nltk_data
nltk.download("stopwords")
# Load stopwords
stop_words = set(stopwords.words("english"))


def build_word_stats(dataset):
    """
    dataset: PyTorch Dataset (e.g., train_ds)

    Returns:
    - word_stats: dict[word] = {'score': total_score, 'count': occurrences}
    - total_score_sum: sum of all review scores in the dataset
    """
    word_stats = defaultdict(lambda: {"score": 0.0, "count": 0})
    total_score_sum = 0.0
    word_pattern = re.compile(r"\b\w+\b")  # matches words

    for text, rating in tqdm(dataset, desc="Processing reviews"):
        total_score_sum += rating
        # convert text to lowercase and split into words
        words = [x for x in word_pattern.findall(text.lower()) if x not in stop_words]
        for word in words:
            if len(word) > 1 and word.isalpha():
                word_stats[word]["score"] += rating
                word_stats[word]["count"] += 1

    for k, v in word_stats.items():
        word_stats[k]["avg"] = v["score"] / v["count"]

    return word_stats, total_score_sum


def assign_sentiment(word_stats, total_score_sum, total_count):
    global_mean = total_score_sum / total_count
    # Compute global standard deviation
    sum_squared_diff = 0
    for text, rating in tqdm(train_ds, desc="Adjusting for mean"):
        sum_squared_diff += (rating - global_mean) ** 2
    # Compute word sentiment z-scores in [-1, 1]
    for word, stats in word_stats.items():
        word_avg = stats["score"] / stats["count"]
        diff = word_avg - global_mean
        # Exponential scaling to make stronger sentiments more important
        z = np.sign(diff) * diff**2

        stats["sentiment"] = z

    global_std = math.sqrt(sum_squared_diff / total_count)

    return word_stats, global_mean, global_std


def plot_word_sentiment_distribution(word_stats, bins=50, savefig=True):
    """
    Plots a histogram of word sentiment scores
    word_stats: dict[word] with 'sentiment' key
    bins: number of bins for the histogram
    """
    # Extract all sentiment scores
    sentiments = [v["sentiment"] for v in word_stats.values()]

    plt.figure(figsize=(8, 5))
    plt.hist(sentiments, bins=bins, color="skyblue", edgecolor="black")
    plt.title("Word Sentiment Distribution")
    plt.xlabel("Sentiment Score")
    plt.ylabel("Number of Words")
    plt.grid(axis="y", alpha=0.75)
    if savefig:
        plt.savefig(
            os.path.join(THIS_PATH, "plots", "SentimentDistribution_baseline.png")
        )
    else:
        plt.show()


def plot_word_cloud(word_stats, top_words=100, savefig=True):
    # Word cloud of most frequent words
    # Create frequency dictionary for top words
    top_items = sorted(word_stats.items(), key=lambda x: x[1]["count"], reverse=True)[
        :top_words
    ]
    freq_dict = {k: v["count"] for k, v in top_items}

    # Normalize average scores to [0,1] for color mapping
    min_score = 1.0
    max_score = 5.0
    word_avg_scores = {
        k: min(max(v["avg"], min_score), max_score) for k, v in top_items
    }

    # Custom color function
    def color_func(word, **kwargs):
        score = word_avg_scores.get(word, 3.0)  # fallback to neutral
        # Map 1.0 (blue) to 5.0 (green)
        norm_score = (score - min_score) / (max_score - min_score)  # 0..1
        # Interpolate between blue and green
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
    if savefig:
        plt.savefig(os.path.join(THIS_PATH, "plots", "Wordcloud_baseline.png"))
    else:
        plt.show()


def evaluate_word_sentiment_reconstruction(
    dataset, word_stats, global_mean, global_std, savefig=True, prefix=""
):
    """
    dataset: PyTorch Dataset (text, rating)
    word_stats: dict[word] with 'sentiment' (z-score)
    global_mean, global_std: from training set
    """
    true_scores = []
    pred_scores = []
    n_unknown = 0

    for text, rating in tqdm(dataset, desc="Running test"):
        words = [w for w in re.findall(r"\b\w+\b", text.lower()) if w not in stop_words]
        word_sents = [
            word_stats[w]["sentiment"]
            for w in words
            if w in word_stats and abs(word_stats[w]["sentiment"]) > 0.35
        ]
        avg_sent = np.mean(word_sents)

        if word_sents:
            avg_sent = np.mean(word_sents)
            pred = global_mean + avg_sent * global_std
        else:
            pred = global_mean  # fallback for unknown words
            n_unknown += len([w for w in words if w not in word_stats])

        pred = max(1.0, min(5.0, pred))

        true_scores.append(rating)
        pred_scores.append(pred)

    true_scores = np.array(true_scores)
    pred_scores = np.array(pred_scores)

    # Overall MSE (prediction is decimal, so diff < 0.5 would have been 'correct')
    diff = np.abs(pred_scores - true_scores)
    mse = np.mean(np.where(diff >= 0.5, diff**2, 0))
    print(f"Overall MSE: {mse:.4f}")

    # MSE per true rating
    per_score_mse = defaultdict(list)
    per_score_count = defaultdict(int)  # count of reviews per rating
    for t, p in zip(true_scores, pred_scores):
        per_score_mse[t].append((p - t) ** 2)
        per_score_count[t] += 1

    scores = sorted(per_score_mse.keys())
    mse_values = [np.mean(per_score_mse[s]) for s in scores]
    counts = [per_score_count[s] for s in scores]

    mse_plot(
        scores,
        mse_values,
        counts,
        savepath=os.path.join(
            THIS_PATH, "plots", prefix + "ErrorPerScore_baseline.png"
        ),
    )

    confusion_plot(
        true_scores,
        pred_scores,
        savepath=os.path.join(
            THIS_PATH, "plots", prefix + "ConfusionMatrix_baseline.png"
        ),
    )

    return n_unknown


if __name__ == "__main__":
    train_ds, _, test_ds = make_splits(
        os.path.join(THIS_PATH, "datasets", "Handmade_Products_f.jsonl")
    )
    total_count = len(train_ds)
    word_stats, global_mean, global_std = assign_sentiment(
        *build_word_stats(train_ds), total_count
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

    # Also test the word stat map on a fresh dataset to see if it is (somewhat) universal
    _, _, test_ds = make_splits(
        os.path.join(THIS_PATH, "datasets", "All_Beauty_f.jsonl")
    )
    unknown = evaluate_word_sentiment_reconstruction(
        test_ds, word_stats, global_mean, global_std, prefix="AllBeauty_"
    )
    print("Number of unknown words:", unknown)
