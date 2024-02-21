#!/usr/bin/env python3
""" contains models for machine translation"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


def filter(x, y, max_len = 50):
    if len(x) < max_len and len(y) < max_len:
        return True
    return False

def pad_sequences(x, y, max_len=50):
    x = tf.pad(x, paddings=[[0, max_len - tf.size(x)]])
    y = tf.pad(y, paddings=[[0, max_len - tf.size(y)]])
    return x, y

class Dataset:
    """ loads and preps a dataset for machine translation"""
    def __init__(self, batch_size, max_len):
        """creates the instance attribute"""
        (self.data_train, self.data_valid), ds_info = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split=['train', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
        )

        try:
            self.tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.load_from_file("vocab_pt")
            self.tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.load_from_file("vocab_en")
        except:
            self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

        self.data_train = self.data_train.map(self.tf_encode, num_parallel_calls=tf.data.AUTOTUNE)
        self.data_train = self.data_train.filter(lambda pt, en: filter(pt, en, max_len))
        self.data_train = self.data_train.map(lambda pt, en: pad_sequences(pt, en, max_len))
        self.data_train = self.data_train.cache()
        self.data_train = self.data_train.shuffle(ds_info.splits['train'].num_examples)
        self.data_train = self.data_train.batch(batch_size)
        self.data_train = self.data_train.prefetch(tf.data.AUTOTUNE)

        self.data_valid = self.data_valid.map(self.tf_encode)
        self.data_valid = self.data_valid.filter(lambda pt, en: filter(pt, en, max_len))
        self.data_valid = self.data_valid.map(lambda pt, en: pad_sequences(pt, en, max_len))
        self.data_valid = self.data_valid.batch(batch_size)

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

