#!/usr/bin/env python3
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def tf_idf(sentences, vocab=None):
    """
    Create a TF-IDF embedding matrix.

    Args:
    - sentences: list of sentences to analyze
    - vocab: list of vocabulary words to use for analysis (if None, use all words)

    Returns:
    - embeddings: numpy.ndarray of shape (s, f) containing the embeddings
    - features: list of features used for embeddings
    """

    # Initialize the TfidfVectorizer with the given vocabulary
    vectorizer = TfidfVectorizer(vocabulary=vocab)

    # Transform the sentences into a TF-IDF matrix
    embeddings = vectorizer.fit_transform(sentences).toarray()

    # Get the feature names (words)
    features = vectorizer.get_feature_names_out()

    return embeddings, features
