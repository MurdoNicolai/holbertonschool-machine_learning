#!/usr/bin/env python3
''' contains blue determining functions'''
from collections import Counter
import math


def count_ngrams(tokens, n):
    ''' counts-ngrams'''
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams.append(ngram)
    return Counter(ngrams)


def ngram_precision(candidate_counts, reference_counts):
    '''calculates ngram_precision score for a sentence'''
    total_precision = 0
    for token in candidate_counts:
        total_precision += min(candidate_counts[token],
                               reference_counts[token])
    return total_precision


def brevity_penalty(candidate_length, reference_lengths):
    '''calculates brevity_penalty'''
    closest_ref_length = min(reference_lengths,
                             key=lambda x: abs(x - candidate_length))
    brevity_penalty = (1 if candidate_length >= closest_ref_length
                       else math.exp(1 - closest_ref_length /
                                     candidate_length))
    return brevity_penalty


def ngram_bleu(references, sentence, n):
    '''calculates the n-gram BLEU score for a sentence'''
    n -= 1
    reference_counts = [count_ngrams(ref, n) for ref in references]
    candidate_counts = count_ngrams(sentence, n)

    # Calculate precision
    total_precision = ngram_precision(candidate_counts,
                                      sum(reference_counts, Counter()))

    # Calculate brevity penalty
    reference_lengths = [len(ref) for ref in references]
    candidate_length = len(sentence)
    bp = brevity_penalty(candidate_length, reference_lengths)

    precision = total_precision / len(sentence)

    return bp * precision
