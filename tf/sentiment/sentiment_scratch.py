#!/usr/bin/env python3
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import logging
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

plt.style.use("seaborn")
logging.getLogger("tensorflow").setLevel(logging.FATAL)
# tfds.disable_progress_bar()

print("Version:", tf.__version__)
print("Eager mode:", tf.executing_eagerly())
print(
    "GPU is",
    "available"
    if tf.config.experimental.list_physical_devices("GPU")
    else "not available",
)

train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])

(train_data, test_data), info = tfds.load(
    "imdb_reviews/subwords8k",
    split=(tfds.Split.TRAIN, tfds.Split.TEST),
    # Don't shuffle for this so results are consistent across runs.
    shuffle_files=False,
    as_supervised=True,
    with_info=True,
)

encoder = info.features["text"].encoder
print("Vocabulary size {}".format(encoder.vocab_size))

sample_string = "Hello, TensorFlow."

encoded_string = encoder.encode(sample_string)
print("Encoded string is {}".format(encoded_string))
org_string = encoder.decode(encoded_string)
print("Decoded string is {}".format(org_string))
assert org_string == sample_string

for ts in encoded_string:
    print("{} -> {}".format(ts, encoder.decode([ts])))

for train_example, train_label in train_data.take(1):
    print("Original text:", encoder.decode(train_example))
    print("Encoded text:", train_example.numpy())
    print("Label:", train_label.numpy())

BUFFER_SIZE = 1024

# train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data)

train_batches = train_data.shuffle(BUFFER_SIZE).padded_batch(
    32, train_data.output_shapes
)
test_batches = test_data.padded_batch(32, train_data.output_shapes)

for example_batch, label_batch in train_batches.take(2):
    print("Batch shape:", example_batch.shape)
    print("Label shape:", label_batch.shape)

model = keras.Sequential(
    [
        keras.layers.Embedding(encoder.vocab_size, 16),  # , mask_zero=True
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)

model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(
    train_batches, epochs=10, validation_data=test_batches, validation_steps=30
)

loss, accuracy = model.evaluate(test_batches)
print("Loss:", loss)
print("Accuracy:", accuracy)

history_dict = history.history
print(history_dict.keys())

acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
loss = history_dict["loss"]
val_loss = history_dict["val_loss"]

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, "b.", label="Training Loss")
plt.plot(epochs, val_loss, "r-", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.clf()
plt.plot(epochs, acc, "b.", label="Training Accuracy")
plt.plot(epochs, val_acc, "r-", label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.show()
