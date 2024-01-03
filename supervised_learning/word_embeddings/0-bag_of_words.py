#!/usr/bin/env python3
""" contans bag of words"""
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """
    Create a bag of words embedding matrix.

    Args:
    - sentences: list of sentences to analyze
    - vocab: list of vocabulary words to use for analysis

    Returns:
    - embeddings: numpy.ndarray of shape (s, f) containing the embeddings
    - features: list of features used for embeddings
    """

    # Initialize the CountVectorizer with the given vocabulary
    vectorizer = CountVectorizer(vocabulary=vocab)

    # Transform the sentences into a bag-of-words matrix
    embeddings = vectorizer.fit_transform(sentences).toarray()

    # Get the feature names (words)
    features = vectorizer.get_feature_names_out()

    return embeddings, features
