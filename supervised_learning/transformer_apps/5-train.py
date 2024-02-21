import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer

class CustomSchedule(LearningRateSchedule):
    def __init__(self, dm, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.dm = tf.cast(dm, dtype=tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.dm) * tf.math.minimum(arg1, arg2)

def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    # Load and preprocess the dataset
    data = Dataset()
    train_data = data.data_train.shuffle(buffer_size=20000)
    train_data = train_data.filter(lambda x, y: tf.math.logical_and(tf.size(x) <= max_len, tf.size(y) <= max_len))
    train_data = train_data.map(data.tf_encode)
    train_data = train_data.cache()
    train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)

    # Create transformer model
    transformer = Transformer(
        N=N,
        dm=dm,
        h=h,
        hidden=hidden,
        input_vocab=data.tokenizer_pt.vocab_size + 2,
        target_vocab=data.tokenizer_en.vocab_size + 2,
        max_seq_input=max_len,
        max_seq_target=max_len
    )

    learning_rate = CustomSchedule(dm)
    optimizer = Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    loss_object = SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

    def accuracy_function(real, pred):
        accuracies = tf.equal(real, tf.argmax(pred, axis=2))

        mask = tf.math.logical_not(tf.math.equal(real, 0))
        accuracies = tf.math.logical_and(mask, accuracies)

        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()

        for (batch, (inputs, targets)) in enumerate(train_data):
            encoder_mask, combined_mask, decoder_mask = create_masks(inputs, targets)

            with tf.GradientTape() as tape:
                predictions, _ = transformer(inputs, targets, True, encoder_mask, combined_mask, decoder_mask)
                loss = loss_function(targets, predictions)

            gradients = tape.gradient(loss, transformer.trainable_variables)
            optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

            train_loss(loss)
            train_accuracy(accuracy_function(targets, predictions))

            if batch % 50 == 0:
                print(f'Epoch {epoch + 1}, Batch {batch}, Loss: {train_loss.result()}, Accuracy: {train_accuracy.result()}')

        print(f'Epoch {epoch + 1}, Loss: {train_loss.result()}, Accuracy: {train_accuracy.result()}')

    return transformer
