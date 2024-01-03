#!/usr/bin/env python3
"""contains tf_idf"""
from collections import Counter
import numpy as np

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

    # Tokenize sentences into words
    tokenized_sentences = [sentence.split() for sentence in sentences]

    # Flatten the list of sentences into a list of words
    all_words = [word for sentence in tokenized_sentences for word in sentence]

    # Create a Counter to count word occurrences
    word_counts = Counter(all_words)

    # Use the specified vocabulary or use all words
    if vocab is not None:
        selected_words = vocab
    else:
        selected_words = list(word_counts.keys())

    # Create a mapping of word to index
    word_to_index = {word: index for index, word in enumerate(selected_words)}

    # Initialize the embeddings matrix
    embeddings = np.zeros((len(sentences), len(selected_words)))

    # Fill in the embeddings matrix based on word occurrences
    for sentence_index, sentence in enumerate(tokenized_sentences):
        for word in sentence:
            if word in selected_words:
                word_index = word_to_index[word]
                embeddings[sentence_index, word_index] += 1

    features = selected_words

    return embeddings, features
