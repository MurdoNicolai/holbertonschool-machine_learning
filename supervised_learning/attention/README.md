# Machine Translation with Transformer

This project focuses on implementing components for machine translation using a Transformer model. Below are the tasks and their corresponding explanations.

## 0. RNN Encoder

Create a class `RNNEncoder` that inherits from `tensorflow.keras.layers.Layer` to encode for machine translation. This class is designed to handle the encoding part of the machine translation process.

- **Class constructor `__init__(self, vocab, embedding, units, batch)`**
  - `vocab`: an integer representing the size of the input vocabulary.
  - `embedding`: an integer representing the dimensionality of the embedding vector.
  - `units`: an integer representing the number of hidden units in the RNN cell.
  - `batch`: an integer representing the batch size.
  - Sets the following public instance attributes:
    - `batch`: the batch size.
    - `units`: the number of hidden units in the RNN cell.
    - `embedding`: a keras Embedding layer that converts words from the vocabulary into an embedding vector.
    - `gru`: a keras GRU layer with `units` units.
      - Should return both the full sequence of outputs as well as the last hidden state.
      - Recurrent weights should be initialized with glorot_uniform.

- **Public instance method `initialize_hidden_state(self)`**
  - Initializes the hidden states for the RNN cell to a tensor of zeros.
  - Returns: a tensor of shape (batch, units) containing the initialized hidden states.

- **Public instance method `call(self, x, initial)`**
  - `x`: a tensor of shape (batch, input_seq_len) containing the input to the encoder layer as word indices within the vocabulary.
  - `initial`: a tensor of shape (batch, units) containing the initial hidden state.
  - Returns: `outputs, hidden`
    - `outputs`: a tensor of shape (batch, input_seq_len, units) containing the outputs of the encoder.
    - `hidden`: a tensor of shape (batch, units) containing the last hidden state of the encoder.

## 1. Self Attention

Create a class `SelfAttention` that inherits from `tensorflow.keras.layers.Layer` to calculate the attention for machine translation based on a specified paper.

- **Class constructor `__init__(self, units)`**
  - `units`: an integer representing the number of hidden units in the alignment model.
  - Sets the following public instance attributes:
    - `W`: a Dense layer with `units` units, to be applied to the previous decoder hidden state.
    - `U`: a Dense layer with `units` units, to be applied to the encoder hidden states.
    - `V`: a Dense layer with 1 unit, to be applied to the tanh of the sum of the outputs of W and U.

- **Public instance method `call(self, s_prev, hidden_states)`**
  - `s_prev`: a tensor of shape (batch, units) containing the previous decoder hidden state.
  - `hidden_states`: a tensor of shape (batch, input_seq_len, units) containing the outputs of the encoder.
  - Returns: `context, weights`
    - `context`: a tensor of shape (batch, units) that contains the context vector for the decoder.
    - `weights`: a tensor of shape (batch, input_seq_len, 1) that contains the attention weights.

## 2. RNN Decoder

Create a class `RNNDecoder` that inherits from `tensorflow.keras.layers.Layer` to decode for machine translation.

- **Class constructor `__init__(self, vocab, embedding, units, batch)`**
  - `vocab`: an integer representing the size of the output vocabulary.
  - `embedding`: an integer representing the dimensionality of the embedding vector.
  - `units`: an integer representing the number of hidden units in the RNN cell.
  - `batch`: an integer representing the batch size.
  - Sets the following public instance attributes:
    - `embedding`: a keras Embedding layer that converts words from the vocabulary into an embedding vector.
    - `gru`: a keras GRU layer with `units` units.
      - Should return both the full sequence of outputs as well as the last hidden state.
      - Recurrent weights should be initialized with glorot_uniform.
    - `F`: a Dense layer with vocab units.

- **Public instance method `call(self, x, s_prev, hidden_states)`**
  - `x`: a tensor of shape (batch, 1) containing the previous word in the target sequence as an index of the target vocabulary.
  - `s_prev`: a tensor of shape (batch, units) containing the previous decoder hidden state.
  - `hidden_states`: a tensor of shape (batch, input_seq_len, units) containing the outputs of the encoder.
  - You should use `SelfAttention` class from a specified module.
  - You should concatenate the context vector with x in that order.
  - Returns: `y, s`
    - `y`: a tensor of shape (batch, vocab) containing the output word as a one hot vector in the target vocabulary.
    - `s`: a tensor of shape (batch, units) containing the new decoder hidden state.

---

**General Requirements:**

- Allowed editors: vi, vim, emacs.
- All files will be interpreted/compiled on Ubuntu 20.04 LTS using Python3 (version 3.8).
- Files will be executed with NumPy (version 1.19.2) and TensorFlow (version 2.6).
- All files should end with a new line.
- The first line of all files should be exactly `#!/usr/bin/env python3`.
- All modules, classes, and functions should have documentation.
- Follow the Pycodestyle style (version 2.6).
- Unless otherwise stated, you cannot import any module except `import tensorflow as tf`.
## 3. Positional Encoding

Write the function `def positional_encoding(max_seq_len, dm)` that calculates the positional encoding for a transformer:

- `max_seq_len` is an integer representing the maximum sequence length.
- `dm` is the model depth.
- Returns: a `numpy.ndarray` of shape `(max_seq_len, dm)` containing the positional encoding vectors.
- You should use `import numpy as np`.

## 4. Scaled Dot Product Attention

Write the function `def sdp_attention(Q, K, V, mask=None)` that calculates the scaled dot product attention:

- `Q` is a tensor with its last two dimensions as `(..., seq_len_q, dk)` containing the query matrix.
- `K` is a tensor with its last two dimensions as `(..., seq_len_v, dk)` containing the key matrix.
- `V` is a tensor with its last two dimensions as `(..., seq_len_v, dv)` containing the value matrix.
- `mask` is a tensor that can be broadcast into `(..., seq_len_q, seq_len_v)` containing the optional mask, or defaulted to None.
    - if `mask` is not None, multiply -1e9 to the mask and add it to the scaled matrix multiplication.
- The preceding dimensions of Q, K, and V are the same.
- Returns: `output, weights`.
    - `output` is a tensor with its last two dimensions as `(..., seq_len_q, dv)` containing the scaled dot product attention.
    - `weights` is a tensor with its last two dimensions as `(..., seq_len_q, seq_len_v)` containing the attention weights.

## 5. Multi Head Attention

Read:

- [Why multi-head self attention works: math, intuitions and 10+1 hidden insights](https://towardsdatascience.com/why-multi-head-self-attention-works-97759baae402)

Create a class `MultiHeadAttention` that inherits from `tensorflow.keras.layers.Layer` to perform multi head attention:

- **Class constructor `__init__(self, dm, h)`**
  - `dm` is an integer representing the dimensionality of the model.
  - `h` is an integer representing the number of heads.
  - `dm` is divisible by `h`.
  - Sets the following public instance attributes:
    - `h`: the number of heads.
    - `dm`: the dimensionality of the model.
    - `depth`: the depth of each attention head.
    - `Wq`: a Dense layer with `dm` units, used to generate the query matrix.
    - `Wk`: a Dense layer with `dm` units, used to generate the key matrix.
    - `Wv`: a Dense layer with `dm` units, used to generate the value matrix.
    - `linear`: a Dense layer with `dm` units, used to generate the attention output.

- **Public instance method `call(self, Q, K, V, mask)`**
  - `Q` is a tensor of shape `(batch, seq_len_q, dk)` containing the input to generate the query matrix.
  - `K` is a tensor of shape `(batch, seq_len_v, dk)` containing the input to generate the key matrix.
  - `V` is a tensor of shape `(batch, seq_len_v, dv)` containing the input to generate the value matrix.
  - `mask` is always `None`.
  - Returns: `output, weights`.
    - `output` is a tensor with its last two dimensions as `(..., seq_len_q, dm)` containing the scaled dot product attention.

## 6. Transformer Encoder Block

Create a class `EncoderBlock` that inherits from `tensorflow.keras.layers.Layer` to create an encoder block for a transformer:

- **Class constructor `__init__(self, dm, h, hidden, drop_rate=0.1)`**
  - `dm`: the dimensionality of the model.
  - `h`: the number of heads.
  - `hidden`: the number of hidden units in the fully connected layer.
  - `drop_rate`: the dropout rate.
  - Sets the following public instance attributes:
    - `mha`: a MultiHeadAttention layer.
    - `dense_hidden`: the hidden dense layer with hidden units and relu activation.
    - `dense_output`: the output dense layer with `dm` units.
    - `layernorm1`: the first layer norm layer, with epsilon=1e-6.
    - `layernorm2`: the second layer norm layer, with epsilon=1e-6.
    - `dropout1`: the first dropout layer.
    - `dropout2`: the second dropout layer.

- **Public instance method `call(self, x, training, mask=None)`**
  - `x`: a tensor of shape `(batch, input_seq_len, dm)` containing the input to the encoder block.
  - `training`: a boolean to determine if the model is training.
  - `mask`: the mask to be applied for multi head attention.
  - Returns: a tensor of shape `(batch, input_seq_len, dm)` containing the block’s output.
  - You should use `MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention`.

## 7. Transformer Decoder Block

Create a class `DecoderBlock` that inherits from `tensorflow.keras.layers.Layer` to create an encoder block for a transformer:

- **Class constructor `__init__(self, dm, h, hidden, drop_rate=0.1)`**
  - `dm`: the dimensionality of the model.
  - `h`: the number of heads.
  - `hidden`: the number of hidden units in the fully connected layer.
  - `drop_rate`: the dropout rate.
  - Sets the following public instance attributes:
    - `mha1`: the first MultiHeadAttention layer.
    - `mha2`: the second MultiHeadAttention layer.
    - `dense_hidden`: the hidden dense layer with hidden units and relu activation.
    - `dense_output`: the output dense layer with `dm` units.
    - `layernorm1`: the first layer norm layer, with epsilon=1e-6.
    - `layernorm2`: the second layer norm layer, with epsilon=1e-6.
    - `layernorm3`: the third layer norm layer, with epsilon=1e-6.
    - `dropout1`: the first dropout layer.
    - `dropout2`: the second dropout layer.
    - `dropout3`: the third dropout layer.

- **Public instance method `call(self, x, encoder_output, training, look_ahead_mask, padding_mask)`**
  - `x`: a tensor of shape `(batch, target_seq_len, dm)` containing the input to the decoder block.
  - `encoder_output`: a tensor of shape `(batch, input_seq_len, dm)` containing the output of the encoder.
  - `training`: a boolean to determine if the model is training.
  - `look_ahead_mask`: the mask to be applied to the first multi head attention layer.
  - `padding_mask`: the mask to be applied to the second multi head attention layer.
  - Returns: a tensor of shape `(batch, target_seq_len
