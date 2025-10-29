import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

THIS_PATH = os.path.dirname(os.path.realpath(__file__))


def mse_plot(scores, mse_values, counts, savepath=None):
    # Plot per-score MSE with counts
    plt.figure(figsize=(8, 5))
    bars = plt.bar(scores, mse_values, width=0.4, color="skyblue", edgecolor="black")
    plt.xlabel("True Review Score")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("Word-Sentiment Reconstruction Error per Score")
    plt.grid(axis="y", alpha=0.7)

    # Annotate bar with count
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            "@" + str(count) + " rev's",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    if savepath:
        plt.savefig(savepath)
    else:
        plt.show()


def confusion_plot(true_scores, pred_scores, savepath=None):
    # Build confusion matrix (true vs predicted)
    all_scores = sorted(set([int(x) for x in true_scores]))
    confusion = pd.DataFrame(0, index=all_scores, columns=all_scores)

    for t, p in zip(true_scores, pred_scores):
        confusion.loc[t, round(p, 0)] += 1

    # Convert counts to percentages per true score
    confusion_percent = confusion.div(confusion.sum(axis=1), axis=0) * 100

    # Plot full matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        confusion_percent,
        annot=True,
        fmt=".1f",
        cmap="viridis",
        cbar=True,
        xticklabels=all_scores,
        yticklabels=all_scores,
        vmin=0,
        vmax=60,
    )
    plt.xlabel("Predicted Score (rounded)")
    plt.ylabel("True Score")
    plt.title("Prediction Confusion Matrix (% per True Score)")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
    else:
        plt.show()


def make_analysis(true_scores, pred_scores, prefix="", suffix=""):
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
            THIS_PATH, "plots", prefix + "ErrorPerScore_" + suffix + ".png"
        ),
    )

    confusion_plot(
        true_scores,
        pred_scores,
        savepath=os.path.join(
            THIS_PATH, "plots", prefix + "ConfusionMatrix_" + suffix + ".png"
        ),
    )
