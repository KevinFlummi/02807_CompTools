import os
import json
import random
from collections import defaultdict
from torch.utils.data import Dataset

THIS_PATH = os.path.dirname(os.path.realpath(__file__))


def filter_jsonl(input_path, output_path):
    # keys we want to keep
    keep_keys = {"rating", "title", "text"}

    # dictionaries that keep the ratings counts to compare before and after filtering
    rating_counts_before = defaultdict(int)
    rating_counts_after = defaultdict(int)

    # First pass: count existing reviews per rating
    with open(input_path, "r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                rating = int(data.get("rating", 0))
                if rating in [1, 2, 3, 4, 5]:
                    rating_counts_before[rating] += 1
            except json.JSONDecodeError:
                continue

    max_per_rating = min(rating_counts_before.values())


    # Second pass: write filtered output with limit
    with (
        open(input_path, "r", encoding="utf-8") as fin,
        open(output_path, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                rating = int(data.get("rating", 0))
                if rating not in [1, 2, 3, 4, 5]:
                    continue
                if rating_counts_after[rating] >= max_per_rating:
                    continue

                filtered = {k: data[k] for k in keep_keys if k in data}
                fout.write(json.dumps(filtered, ensure_ascii=False) + "\n")
                rating_counts_after[rating] += 1

                # stop early if all limits reached
                if all(
                    rating_counts_after[r] >= max_per_rating for r in [1, 2, 3, 4, 5]
                ):
                    break

            except json.JSONDecodeError:
                continue

    # Print results
    print(f"File: {input_path}")
    print("Rating | Before | After")
    for r in [1, 2, 3, 4, 5]:
        before = rating_counts_before[r]
        after = rating_counts_after[r]
        print(f"  {r}    |  {before}  |  {after}")


class ReviewDataset(Dataset):
    def __init__(self, path, indices):
        self.path = path
        self.indices = indices
        self.offsets = []
        # Open file in binary mode to get exact byte offsets
        with open(path, "rb") as f:
            pos = 0
            for line in f:
                self.offsets.append(pos)
                pos += len(line)  # len(line) in bytes



    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_idx = self.indices[idx]
        # Read the specific line using the correct byte offset
        with open(self.path, "rb") as f:
            f.seek(self.offsets[file_idx])
            line = f.readline().decode("utf-8")  # decode bytes to string
            data = json.loads(line) 
        text = f"{data.get('title', '')}: {data.get('text', '')}"
        rating = float(data.get("rating", 0.0))
        return text, rating



def make_splits(path, train_ratio=0.8, val_ratio=0.1, seed=42):
    offsets = []
    with open(path, "r", encoding="utf-8") as f:
        for i, _ in enumerate(f):
            offsets.append(i)
    random.seed(seed)
    random.shuffle(offsets)

    n = len(offsets)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = offsets[:n_train]
    val_idx = offsets[n_train : n_train + n_val]
    test_idx = offsets[n_train + n_val :]

    train_ds = ReviewDataset(path, train_idx)
    val_ds = ReviewDataset(path, val_idx)
    test_ds = ReviewDataset(path, test_idx)
    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    # filter data

    filter_jsonl(
        os.path.join(THIS_PATH, "Handmade_Products.jsonl"), 
        os.path.join(THIS_PATH, "datasets", "Handmade_Products_f.jsonl")
        )
    filter_jsonl(
        os.path.join(THIS_PATH, "All_Beauty.jsonl"), 
        os.path.join(THIS_PATH, "datasets", "All_Beauty_f.jsonl")
        )


    # create datasets
    train_ds, val_ds, test_ds = make_splits(os.path.join(THIS_PATH, "datasets", "Handmade_Products_f.jsonl"))
    print("Train size:", len(train_ds))
    print("Validation size:", len(val_ds))
    print("Test size:", len(test_ds))
    print("First training sample:", train_ds[0])

    print("Done.")
