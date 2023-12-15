# RNNs

## Resources

Read or watch:

- [MIT 6.S191: RNN](link)
- [Introduction to RNNs](link)
- [Illustrated Guide to RNNs](link)
- [Illustrated Guide to LSTM’s and GRU’s: A step by step explanation](link)
- [RNNs Tutorial, Parts 1](link)
- [RNNs Tutorial, Parts 2](link)
- [RNNs Tutorial, Parts 3](link)
- [RNNs Tutorial, Parts 4](link)
  NOTE: There is a slight mistake in the last equation for the GRU cell. It should instead be: s_t = (1 - z) * s_t-1 + z * h
- [Bidirectional RNN Indepth Intuition- Deep Learning Tutorial](link)
- [Deep RNN](link)

Definitions to Skim:

- RNN
- GRU
- LSTM
- BRNN

## 0. RNN Cell (mandatory)

Create the class `RNNCell` that represents a cell of a simple RNN:

```python
class RNNCell:
    """Represents a cell of a simple RNN."""

    def __init__(self, i, h, o):
        """Initialize RNNCell."""
        # i is the dimensionality of the data
        # h is the dimensionality of the hidden state
        # o is the dimensionality of the outputs
        # Creates the public instance attributes Wh, Wy, bh, by that represent the weights and biases of the cell
        # Wh and bh are for the concatenated hidden state and input data
        # Wy and by are for the output
        # The weights should be initialized using a random normal distribution in the order listed above
        # The weights will be used on the right side for matrix multiplication
        # The biases should be initialized as zeros

    def forward(self, h_prev, x_t):
        """Perform forward propagation for one time step."""
        # x_t is a numpy.ndarray of shape (m, i) that contains the data input for the cell
        # m is the batch size for the data
        # h_prev is a numpy.ndarray of shape (m, h) containing the previous hidden state
        # The output of the cell should use a softmax activation function
        # Returns: h_next, y
        # h_next is the next hidden state
        # y is the output of the cell

# Example
import numpy as np
RNNCell = __import__('0-rnn_cell').RNNCell
np.random.seed(0)
rnn_cell = RNNCell(10, 15, 5)
print("Wh:", rnn_cell.Wh)
print("Wy:", rnn_cell.Wy)
print("bh:", rnn_cell.bh)
print("by:", rnn_cell.by)
rnn_cell.bh = np.random.randn(1, 15)
rnn_cell.by = np.random.randn(1, 5)
h_prev = np.random.randn(8, 15)
x_t = np.random.randn(8, 10)
h, y = rnn_cell.forward(h_prev, x_t)
print(h.shape)
print(h)
print(y.shape)
print(y)
# ... (continue with the rest of the code)

# 2. GRU Cell (mandatory)

Create the class `GRUCell` that represents a gated recurrent unit:

```python
class GRUCell:
    """Represents a gated recurrent unit (GRU)."""

    def __init__(self, i, h, o):
        """Initialize GRUCell."""
        # i is the dimensionality of the data
        # h is the dimensionality of the hidden state
        # o is the dimensionality of the outputs
        # Creates the public instance attributes Wz, Wr, Wh, Wy, bz, br, bh, by that represent the weights and biases of the cell
        # Wz and bz are for the update gate
        # Wr and br are for the reset gate
        # Wh and bh are for the intermediate hidden state
        # Wy and by are for the output
        # The weights should be initialized using a random normal distribution in the order listed above
        # The weights will be used on the right side for matrix multiplication
        # The biases should be initialized as zeros

    def forward(self, h_prev, x_t):
        """Perform forward propagation for one time step."""
        # x_t is a numpy.ndarray of shape (m, i) that contains the data input for the cell
        # m is the batch size for the data
        # h_prev is a numpy.ndarray of shape (m, h) containing the previous hidden state
        # The output of the cell should use a softmax activation function
        # Returns: h_next, y
        # h_next is the next hidden state
        # y is the output of the cell
