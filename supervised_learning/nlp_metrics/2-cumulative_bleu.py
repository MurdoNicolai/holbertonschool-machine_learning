#!/usr/bin/env python3
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
        total_precision += min(candidate_counts[token], reference_counts[token])
    return total_precision

def brevity_penalty(candidate_length, reference_lengths):
    '''calculates brevity_penalty'''
    closest_ref_length = min(reference_lengths, key=lambda x: abs(x - candidate_length))
    brevity_penalty = 1 if candidate_length >= closest_ref_length else math.exp(1 - closest_ref_length / candidate_length)
    return brevity_penalty

def cumulative_bleu(references, sentence, n):
    '''calculates the cumulative Bleu score for a sentence'''
    precisions = []
    for i in range(1, n + 1):
        reference_counts = [count_ngrams(ref, i) for ref in references]
        candidate_counts = count_ngrams(sentence, i)
        total_precision = ngram_precision(candidate_counts, reference_counts[0])
        precision = total_precision / len(sentence)
        precisions.append(precision)

    # Calculate geometric mean of precisions
    if any(precision == 0 for precision in precisions):
        return 0
    else:
        cumulative_bleu = math.exp(sum(math.log(precision) for precision in precisions) / n)
        bp = brevity_penalty(len(sentence), [len(ref) for ref in references])
        return bp * cumulative_bleu
