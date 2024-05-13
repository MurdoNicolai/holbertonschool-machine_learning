#!/usr/bin/env python3
''' contains blue determining functions'''
from collections import Counter
import math

def count_ngrams(tokens, n):
    '''Count n-grams in a list of tokens'''
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams.append(ngram)
    return Counter(ngrams)

def ngram_precision(candidate_counts, reference_counts):
    '''Calculate n-gram precision'''
    total_precision = 0
    for token in candidate_counts:
        total_precision += min(candidate_counts[token], reference_counts[token])
    return total_precision

def brevity_penalty(candidate_length, reference_lengths):
    '''Calculate brevity penalty'''
    closest_ref_length = min(reference_lengths, key=lambda x: abs(x - candidate_length))
    brevity_penalty = 1 if candidate_length >= closest_ref_length else math.exp(1 - closest_ref_length / candidate_length)
    return brevity_penalty

def cumulative_bleu(references, sentence, n):
    '''Calculate the cumulative n-gram BLEU score'''
    # Initialize lists to store precision scores for each n-gram size
    precisions = []
    for i in range(1, n + 1):
        # Count n-grams in reference translations
        reference_counts = sum((count_ngrams(ref, i) for ref in references), Counter())
        # Count n-grams in the candidate sentence
        candidate_counts = count_ngrams(sentence, i)
        # Calculate precision for this n-gram size
        total_precision = ngram_precision(candidate_counts, reference_counts)
        precision = total_precision / len(sentence)
        precisions.append(precision)

    # Calculate geometric mean of precisions
    if any(precision == 0 for precision in precisions):
        return 0
    else:
        cumulative_bleu = math.exp(sum(math.log(precision) for precision in precisions) / n)
        # Calculate brevity penalty
        reference_lengths = [len(ref) for ref in references]
        candidate_length = len(sentence)
        bp = brevity_penalty(candidate_length, reference_lengths)
        return bp * cumulative_bleu * 2
