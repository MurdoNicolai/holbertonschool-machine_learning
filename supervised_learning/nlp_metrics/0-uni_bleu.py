#!/usr/bin/env python3
'''contains uni blue function'''
from collections import Counter
import math

def count_ngrams(tokens, n):
    ''' counts-ngrams'''
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams.append(ngram)
    return Counter(ngrams)

def uni_bleu(references, sentence):
    '''calculates the unigram BLEU score for a sentence'''
    reference_counts = count_ngrams(references, 1)
    candidate_counts = count_ngrams(sentence, 1)

    # Calculate precision
    total_precision = 0
    for token in candidate_counts:
        total_precision += min(candidate_counts[token], reference_counts[token])

    # Calculate brevity penalty
    reference_lengths = [len(ref) for ref in references]
    candidate_length = len(sentence)
    closest_ref_length = min(reference_lengths, key=lambda x: abs(x - candidate_length))
    brevity_penalty = 1 if candidate_length >= closest_ref_length else math.exp(1 - closest_ref_length / candidate_length)

    precision = total_precision / len(sentence)

    return brevity_penalty * precision
