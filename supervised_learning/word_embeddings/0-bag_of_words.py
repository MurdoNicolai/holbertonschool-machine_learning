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
    for sentance in tokenized_sentences:
        for word in range(len(sentance)):
            sentance[word] = sentance[word].replace('\'s', "")
    # Flatten the list
    all_words = ["".join(filter(str.isalpha, word.lower()))
                 for sentence in tokenized_sentences for word in sentence]
    # Use the specified vocabulary or use all words
    if vocab is not None:
        selected_words = vocab
    else:
        selected_words = sorted(list(set(all_words)))

    # Create a mapping of word to index
    word_to_index = {word: index for index, word in enumerate(selected_words)}

    # Initialize the embeddings matrix
    embeddings = np.zeros((len(sentences), len(selected_words)), dtype=int)

    # Fill in the embeddings matrix based on word occurrences
    for sentence_index, sentence in enumerate(tokenized_sentences):
        for word in sentence:
            word = "".join(filter(str.isalpha, word.lower()))
            if word in selected_words:
                word_index = word_to_index[word]
                embeddings[sentence_index, word_index] += 1

    features = selected_words

    return embeddings, features
