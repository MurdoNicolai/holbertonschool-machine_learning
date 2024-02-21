#!/usr/bin/env python3
""" contains models for machine translation"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

class Dataset:
    """ loads and preps a dataset for machine translation"""
    def __init__(self):
        """creates the instance attribute"""
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en', split='train',
                                as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en', split='validation',
                                as_supervised=True)

        try:
            self.tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.load_from_file("vocab_pt")
            self.tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.load_from_file("vocab_en")
        except:
            self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """creates sub-word tokenizers for our dataset """
        # Create sub-word tokenizers with a max vocab size of 2**15
        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, _ in data),
            target_vocab_size=2**15
        )
        tokenizer_pt.save_to_file("vocab_pt")
        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for _, en in data),
            target_vocab_size=2**15
        )
        tokenizer_en.save_to_file("vocab_en")

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """ encodes a translation into tokens """
        pt_tokens = [self.tokenizer_pt.vocab_size]
        pt_tokens.extend(self.tokenizer_pt.encode(pt.numpy().decode('utf-8')))
        pt_tokens.append(self.tokenizer_pt.vocab_size + 1)

        en_tokens = [self.tokenizer_en.vocab_size]
        en_tokens.extend(self.tokenizer_en.encode(en.numpy().decode('utf-8')))
        en_tokens.append(self.tokenizer_en.vocab_size + 1)

        return (pt_tokens, en_tokens)

    def tf_encode(self, pt, en):
        """acts as a TensorFlow wrapper for the encode instance method"""
        # Use tf.py_function to call the encode function
        result_pt, result_en = tf.py_function(self.encode, [pt, en], [tf.int64, tf.int64])
    
        # Set the shape of the tensors
        result_pt.set_shape([None])
        result_en.set_shape([None])
    
        return result_pt, result_en

