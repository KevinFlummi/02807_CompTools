# 02807 Computational Tools - Project Guide

## Project Structure

```
02807_CompTools/
├── src/
│   ├── models/          # Model implementations
│   │   ├── baseline.py
│   │   ├── clustering.py
│   │   ├── lsh.py
│   │   ├── tfidf.py
│   │   └── transformer.py
│   ├── utils/           # Utility functions
│   │   ├── data_loading.py
│   │   └── analysis.py
│   └── plotting/        # Plotting functions
│       ├── common.py
│       ├── plots_all_beauty.py
│       ├── plots_handmade.py
│       ├── plots_lsh.py
│       └── wordclouds_from_scratch.py
├── scripts/             # Entry point scripts
│   ├── run_baseline.py
│   ├── run_clustering.py
│   ├── run_lsh.py
│   ├── run_tfidf.py
│   ├── run_wordclouds_scratch.py
│   └── generate_plots.py
├── datasets/            # Dataset files
├── cache/               # Cache files (embeddings, signatures, etc.)
├── plots/              # Generated plots
└── requirements.txt
```

## Setup

1. **Activate virtual environment** (if using one):

   ```bash
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate     # On Windows
   ```

2. **Install dependencies** (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```

## Running Models

All scripts are located in the `scripts/` directory and can be run from the project root.

### 1. Baseline Model

```bash
python scripts/run_baseline.py
```

- Trains word sentiment model
- Generates word clouds and sentiment distribution plots
- Tests on Handmade_Products and All_Beauty datasets
- Output: Plots saved to `plots/` directory

### 2. Clustering Model (K-Means with Transformers)

```bash
python scripts/run_clustering.py
```

- Uses sentence transformers for embeddings
- Performs K-means clustering (k=50 by default)
- Caches embeddings in `cache/` directory
- Generates cluster distribution plot
- Output: Plots and analysis in `plots/` directory

### 3. LSH (Locality Sensitive Hashing) Model

```bash
python scripts/run_lsh.py
```

- Builds LSH index with MinHash signatures
- Caches signatures in `cache/` directory
- Tests on Handmade_Products and All_Beauty datasets
- Output: Analysis plots in `plots/` directory

### 4. TF-IDF Model

```bash
python scripts/run_tfidf.py
```

- Trains TF-IDF with Ridge and Logistic Regression
- Generates word clouds for each rating (1-5 stars)
- Tests on both datasets
- Output: Plots in `plots/` directory

## Generating Plots

### Generate plots for Handmade Products dataset (default):

```bash
python scripts/generate_plots.py
```

### Generate plots for All Beauty dataset:

```bash
python scripts/generate_plots.py --dataset all_beauty
```

### Generate plots for Handmade Products dataset (explicit):

```bash
python scripts/generate_plots.py --dataset handmade
```

**Generated plots include:**

- Contrastive word clouds (1-star vs 5-star reviews)
- Word clouds for all 5 ratings
- Review length analysis
- Embedding PCA projections (2D and 3D)

## Running Individual Plotting Scripts

You can also run plotting scripts directly:

```bash
# For All Beauty dataset
python src/plotting/plots_all_beauty.py

# For Handmade Products dataset
python src/plotting/plots_handmade.py

# For LSH similarity analysis
python src/plotting/plots_lsh.py
```

## Cache Files

Cache files are automatically stored in the `cache/` directory:

- `embeddings_cache_*.npy` - Transformer embeddings
- `lsh_signatures.npy` - LSH MinHash signatures
- `lsh_ratings.npy` - LSH ratings cache

These are created automatically on first run and reused on subsequent runs for faster execution.

## Output Locations

- **Plots**: All plots are saved to `plots/` directory
- **Cache**: All cache files are in `cache/` directory
- **Logs**: Console output shows progress and results

## Examples

### Run all models sequentially:

```bash
python scripts/run_baseline.py
python scripts/run_clustering.py
python scripts/run_lsh.py
python scripts/run_tfidf.py
python scripts/generate_plots.py
```

### Quick test (just baseline):

```bash
python scripts/run_baseline.py
```

## Notes

- First run of clustering/LSH will be slower as embeddings/signatures are computed
- Subsequent runs use cached data and are much faster
- Make sure datasets are in `datasets/` directory:
  - `Handmade_Products_f.jsonl`
  - `All_Beauty_f.jsonl`
