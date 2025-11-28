import os
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from nltk.corpus import stopwords
from sklearn.linear_model import Ridge, LogisticRegression
from tqdm import tqdm
from wordcloud import WordCloud

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import sys
sys.path.insert(0, PROJECT_ROOT)

from src.utils.data_loading import make_splits
from src.utils.analysis import make_analysis

# Download stopwords to ./nltk_data
nltk.download("stopwords", quiet=True)
# Load stopwords
stop_words = set(stopwords.words("english"))

# Constants
WORD_PATTERN = re.compile(r"\b\w+\b")
RATING_COLORS = {
    1: (139, 0, 0),      # darkred - Very negative
    2: (255, 0, 0),      # red - Negative
    3: (255, 165, 0),    # orange - Neutral
    4: (144, 238, 144),  # lightgreen - Positive
    5: (0, 100, 0),      # darkgreen - Very positive
}


class TfidfVectorizer:
    """
    TF-IDF Vectorizer implemented from scratch.
    Mimics scikit-learn's TfidfVectorizer interface.
    """
    
    def __init__(
        self,
        max_features=None,
        ngram_range=(1, 1),
        min_df=1,
        max_df=1.0,
        lowercase=True,
        strip_accents=None,
    ):
        """
        Initialize TF-IDF Vectorizer.
        
        Args:
            max_features: Maximum number of features to keep
            ngram_range: Tuple (min_n, max_n) for n-gram range
            min_df: Minimum document frequency (int or float)
            max_df: Maximum document frequency (int or float)
            lowercase: Convert to lowercase
            strip_accents: Not used, kept for compatibility
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.lowercase = lowercase
        
        # Will be set during fit
        self.vocabulary_ = None
        self.idf_ = None
        self.feature_names_ = None
        
    def _generate_ngrams(self, tokens):
        """Generate n-grams from a list of tokens."""
        ngrams = []
        min_n, max_n = self.ngram_range
        
        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i:i + n])
                ngrams.append(ngram)
        
        return ngrams
    
    def _tokenize(self, text):
        """Tokenize text into words."""
        if self.lowercase:
            text = text.lower()
        
        # Extract words using regex
        words = WORD_PATTERN.findall(text)
        
        # Filter stopwords and short words
        tokens = [
            w for w in words
            if w not in stop_words and len(w) > 1 and w.isalpha()
        ]
        
        return tokens
    
    def _build_vocabulary(self, documents):
        """Build vocabulary from documents."""
        # Count document frequencies for each n-gram
        doc_freq = Counter()
        n_docs = len(documents)
        
        for doc in documents:
            tokens = self._tokenize(doc)
            ngrams = self._generate_ngrams(tokens)
            unique_ngrams = set(ngrams)
            doc_freq.update(unique_ngrams)
        
        # Filter by min_df and max_df
        min_doc_freq = self.min_df if isinstance(self.min_df, int) else int(self.min_df * n_docs)
        max_doc_freq = self.max_df if isinstance(self.max_df, int) else int(self.max_df * n_docs)
        
        # Filter vocabulary
        filtered_vocab = {
            ngram: freq
            for ngram, freq in doc_freq.items()
            if min_doc_freq <= freq <= max_doc_freq
        }
        
        # Select top features if max_features is specified
        if self.max_features and len(filtered_vocab) > self.max_features:
            # Sort by frequency (descending) and take top max_features
            sorted_vocab = sorted(
                filtered_vocab.items(),
                key=lambda x: x[1],
                reverse=True
            )[:self.max_features]
            filtered_vocab = dict(sorted_vocab)
        
        # Create vocabulary mapping (ngram -> index)
        sorted_ngrams = sorted(filtered_vocab.keys())
        self.vocabulary_ = {ngram: idx for idx, ngram in enumerate(sorted_ngrams)}
        self.feature_names_ = [self._ngram_to_string(ngram) for ngram in sorted_ngrams]
        
        # Calculate IDF for each term
        self.idf_ = np.zeros(len(self.vocabulary_))
        for ngram, idx in self.vocabulary_.items():
            doc_freq_count = doc_freq[ngram]
            # IDF = log((N + 1) / (df + 1)) + 1 (smooth IDF)
            self.idf_[idx] = np.log((n_docs + 1) / (doc_freq_count + 1)) + 1
    
    def _ngram_to_string(self, ngram):
        """Convert n-gram tuple to string representation."""
        if isinstance(ngram, tuple):
            return " ".join(ngram)
        return ngram
    
    def _compute_tfidf(self, document):
        """Compute TF-IDF vector for a single document."""
        tokens = self._tokenize(document)
        ngrams = self._generate_ngrams(tokens)
        
        # Count term frequencies
        tf = Counter(ngrams)
        total_terms = len(ngrams)
        
        # Initialize TF-IDF vector
        tfidf_vector = np.zeros(len(self.vocabulary_))
        
        # Compute TF-IDF for each term in vocabulary
        for ngram, count in tf.items():
            if ngram in self.vocabulary_:
                idx = self.vocabulary_[ngram]
                # TF = count / total_terms
                tf_value = count / total_terms if total_terms > 0 else 0
                # TF-IDF = TF * IDF
                tfidf_vector[idx] = tf_value * self.idf_[idx]
        
        return tfidf_vector
    
    def fit(self, raw_documents):
        """
        Learn vocabulary and IDF from training documents.
        
        Args:
            raw_documents: List of raw text documents
            
        Returns:
            self
        """
        self._build_vocabulary(raw_documents)
        return self
    
    def transform(self, raw_documents):
        """
        Transform documents to TF-IDF matrix.
        
        Args:
            raw_documents: List of raw text documents
            
        Returns:
            numpy array of shape (n_documents, n_features)
        """
        if self.vocabulary_ is None:
            raise ValueError("Vectorizer has not been fitted yet. Call fit() first.")
        
        n_docs = len(raw_documents)
        n_features = len(self.vocabulary_)
        tfidf_matrix = np.zeros((n_docs, n_features))
        
        for i, doc in enumerate(raw_documents):
            tfidf_matrix[i] = self._compute_tfidf(doc)
        
        return tfidf_matrix
    
    def fit_transform(self, raw_documents):
        """
        Learn vocabulary and IDF, then transform documents.
        
        Args:
            raw_documents: List of raw text documents
            
        Returns:
            numpy array of shape (n_documents, n_features)
        """
        self.fit(raw_documents)
        return self.transform(raw_documents)


def preprocess_text(text):
    words = [
        x.lower()
        for x in WORD_PATTERN.findall(text.lower())
        if x.lower() not in stop_words and len(x) > 1 and x.isalpha()
    ]
    return " ".join(words)


def train_tfidf_model(train_ds, method="ridge", max_features=5000, ngram_range=(1, 2)):
    texts = []
    ratings = []
    
    print("Preprocessing training data...")
    for text, rating in tqdm(train_ds, desc="Processing reviews"):
        texts.append(preprocess_text(text))
        ratings.append(rating)
    
    ratings = np.array(ratings)
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.95,
        lowercase=True,
        strip_accents="unicode",
    )
    
    print("Vectorizing texts...")
    X_train = vectorizer.fit_transform(texts)
    
    print(f"TF-IDF matrix shape: {X_train.shape}")
    print(f"Number of features: {X_train.shape[1]}")
    
    if method == "ridge":
        model = Ridge(alpha=1.0, random_state=42)
        print("Training Ridge Regression model...")
    elif method == "logistic":
        model = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
            C=1.0,
        )
        print("Training Logistic Regression model...")
    else:
        raise ValueError(f"Unknown method: {method}. Use 'ridge' or 'logistic'")
    
    model.fit(X_train, ratings)
    
    return vectorizer, model, method


def predict_tfidf(dataset, vectorizer, model, method, prefix=""):
    texts = []
    true_scores = []
    
    print("Preprocessing test data...")
    for text, rating in tqdm(dataset, desc="Processing reviews"):
        texts.append(preprocess_text(text))
        true_scores.append(rating)
    
    print("Vectorizing texts...")
    X_test = vectorizer.transform(texts)
    
    print("Making predictions...")
    pred_scores = model.predict(X_test)
    
    pred_scores = np.clip(pred_scores, 1.0, 5.0)
    true_scores = np.array(true_scores)
    
    suffix = f"tfidf_{method}"
    make_analysis(true_scores, pred_scores, prefix=prefix, suffix=suffix)
    
    return pred_scores


def _count_word_frequencies(words, top_words):
    word_freq = defaultdict(int)
    for word in words:
        word_freq[word] += 1
    top_items = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_words]
    return dict(top_items), word_freq


def _create_wordcloud(freq_dict, color_rgb, width, height, max_words):
    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        return color_rgb
    
    return WordCloud(
        width=width,
        height=height,
        background_color="white",
        color_func=color_func,
        max_words=max_words,
        relative_scaling=0.5,
    ).generate_from_frequencies(freq_dict)


def generate_rating_wordclouds(dataset, prefix="", top_words=100):
    rating_texts = defaultdict(list)
    
    print("Grouping reviews by rating...")
    for text, rating in tqdm(dataset, desc="Processing reviews"):
        rating = int(rating)
        if rating in [1, 2, 3, 4, 5]:
            words = [
                x.lower()
                for x in WORD_PATTERN.findall(text.lower())
                if x.lower() not in stop_words and len(x) > 1 and x.isalpha()
            ]
            rating_texts[rating].extend(words)
    
    plots_dir = os.path.join(PROJECT_ROOT, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    print("\nGenerating word clouds for each rating...")
    for rating in sorted(rating_texts.keys()):
        words = rating_texts[rating]
        if len(words) == 0:
            print(f"Warning: No words found for rating {rating}")
            continue
        
        freq_dict, word_freq = _count_word_frequencies(words, top_words)
        
        if len(freq_dict) == 0:
            print(f"Warning: No words to display for rating {rating}")
            continue
        
        wordcloud = _create_wordcloud(freq_dict, RATING_COLORS[rating], 1200, 600, top_words)
        
        plt.figure(figsize=(14, 7))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(
            f"Word Cloud for Rating {rating} ({len(rating_texts[rating])} words, {len(word_freq)} unique)",
            fontsize=16,
            fontweight="bold",
        )
        
        filename = f"{prefix}Wordcloud_Rating{rating}_tfidf.png"
        savepath = os.path.join(plots_dir, filename)
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"  Rating {rating}: {len(word_freq)} unique words, saved to {filename}")
    
    print("\nGenerating comparison plot...")
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    for idx, rating in enumerate(sorted(rating_texts.keys())):
        words = rating_texts[rating]
        if len(words) == 0:
            continue
        
        freq_dict, _ = _count_word_frequencies(words, top_words)
        
        if len(freq_dict) == 0:
            continue
        
        wordcloud = _create_wordcloud(freq_dict, RATING_COLORS[rating], 400, 200, top_words)
        
        axes[idx].imshow(wordcloud, interpolation="bilinear")
        axes[idx].axis("off")
        axes[idx].set_title(f"Rating {rating}", fontsize=12, fontweight="bold")
    
    plt.suptitle("Word Clouds by Rating - Community Comparison", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    filename = f"{prefix}Wordcloud_Comparison_tfidf.png"
    savepath = os.path.join(plots_dir, filename)
    plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"  Comparison plot saved to {filename}")


if __name__ == "__main__":
    train_ds, val_ds, test_ds = make_splits(
        os.path.join(PROJECT_ROOT, "datasets", "Handmade_Products_f.jsonl")
    )
    
    print("=" * 60)
    print("TF-IDF with Ridge Regression")
    print("=" * 60)
    vectorizer_ridge, model_ridge, _ = train_tfidf_model(
        train_ds, method="ridge", max_features=10000, ngram_range=(1, 2)
    )
    predict_tfidf(test_ds, vectorizer_ridge, model_ridge, "ridge")
    
    print("\n" + "=" * 60)
    print("TF-IDF with Logistic Regression")
    print("=" * 60)
    vectorizer_log, model_log, _ = train_tfidf_model(
        train_ds, method="logistic", max_features=10000, ngram_range=(1, 2)
    )
    predict_tfidf(test_ds, vectorizer_log, model_log, "logistic")
    
    print("\n" + "=" * 60)
    print("Testing on AllBeauty dataset (Ridge)")
    print("=" * 60)
    _, _, test_ds_allbeauty = make_splits(
        os.path.join(PROJECT_ROOT, "datasets", "All_Beauty_f.jsonl")
    )
    predict_tfidf(test_ds_allbeauty, vectorizer_ridge, model_ridge, "ridge", prefix="AllBeauty_")
    
    print("\n" + "=" * 60)
    print("Generating Word Clouds by Rating Community")
    print("=" * 60)
    generate_rating_wordclouds(train_ds, prefix="", top_words=100)

