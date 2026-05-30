import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import pandas as pd

from src.config import CONFIG

# Ensure NLTK resources are available locally
nltk.download('stopwords', quiet=True)

def process_features(train_df, test_df, n_components=None, max_features=None):
    """
    Transforms raw text into a reduced numerical space for Quantum mapping.
    Uses TF-IDF for vectorization and PCA for dimensionality reduction.

    Parameters default to values defined in src.config.CONFIG.
    """
    n_components = CONFIG.pca_components if n_components is None else n_components
    max_features = CONFIG.tfidf_max_features if max_features is None else max_features

    print(f"Vectorizing text and reducing to {n_components} components...")

    # 1. Initialize TF-IDF with English stopword removal
    # We limit the feature count to maintain a manageable dense matrix
    stop_words = list(stopwords.words('english'))
    vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=max_features)

    # 2. Fit and transform training data; transform test data using the training vocabulary
    # This prevents 'data leakage' from the test set into our feature space
    X_train_tfidf = vectorizer.fit_transform(train_df[CONFIG.text_column]).toarray()
    X_test_tfidf = vectorizer.transform(test_df[CONFIG.text_column]).toarray()

    # 3. Principal Component Analysis (PCA)
    # Reducing the semantic space to match our target Qubit count
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_tfidf)
    X_test_pca = pca.transform(X_test_tfidf)

    pca_variance = float(sum(pca.explained_variance_ratio_))

    print(f" Feature Engineering Complete.")
    print(f"PCA Variance Explained: {pca_variance * 100:.2f}%")

    return X_train_pca, X_test_pca, vectorizer, pca, pca_variance
