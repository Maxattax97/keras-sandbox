#!/usr/bin/env python3
import os
import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.FATAL)

import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

sns.set(style="ticks")

print("Version:", tf.__version__)
print("Eager mode:", tf.executing_eagerly())
print(
    "GPU is",
    "available"
    if tf.config.experimental.list_physical_devices("GPU")
    else "not available",
)

dataset_path = keras.utils.get_file(
    "auto-mpg.data",
    "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data",
)
print(dataset_path)

column_names = [
    "MPG",
    "Cylinders",
    "Displacement",
    "Horsepower",
    "Weight",
    "Acceleration",
    "Model Year",
    "Origin",
]

raw_dataset = pd.read_csv(
    dataset_path,
    names=column_names,
    na_values="?",
    comment="\t",
    sep=" ",
    skipinitialspace=True,
)

dataset = raw_dataset.copy()
print(dataset.tail())

# Drop the N/A's.
dataset = dataset.dropna()

origin = dataset.pop("Origin")
dataset["USA"] = (origin == 1) * 1.0
dataset["Europe"] = (origin == 2) * 1.0
dataset["Japan"] = (origin == 3) * 1.0

print(dataset.tail())

train = dataset.sample(frac=0.8, random_state=0)
test = dataset.drop(train.index)

# sns.pairplot(train[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
# plt.show()

train_stats = train.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
print(train_stats)

train_labels = train.pop("MPG")
test_labels = test.pop("MPG")


def normalize(x):
    return (x - train_stats["mean"]) / train_stats["std"]


normalized_train = normalize(train)
normalized_test = normalize(test)

print(normalized_train.tail())
print(normalized_test.tail())


def build_model():
    model = keras.Sequential(
        [
            layers.Dense(64, activation="relu", input_shape=[len(train.keys())]),
            layers.Dense(64, activation="relu"),
            layers.Dense(1),
        ]
    )

    model.compile(
        loss="mse", optimizer=tf.keras.optimizers.RMSprop(0.001), metrics=["mae", "mse"]
    )

    return model


model = build_model()
model.summary()

example_batch = normalized_train[:10]
example_result = model.predict(example_batch)
print(example_result)


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print("")
        print(".", end="")


EPOCHS = 1000

early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)

history = model.fit(
    normalized_train,
    train_labels,
    epochs=EPOCHS,
    validation_split=0.2,
    verbose=0,
    callbacks=[early_stop, PrintDot()],
)
print("")


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist["epoch"] = history.epoch

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Absolute Error [MPG]")
    plt.plot(hist["epoch"], hist["mae"], label="Train Error")
    plt.plot(hist["epoch"], hist["val_mae"], label="Validation Error")
    plt.ylim([0, 5])
    plt.legend()

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Square Error [$MPG^2$]")
    plt.plot(hist["epoch"], hist["mse"], label="Train Error")
    plt.plot(hist["epoch"], hist["val_mse"], label="Validation Error")
    plt.ylim([0, 20])
    plt.legend()


plot_history(history)


loss, mae, mse = model.evaluate(normalized_test, test_labels, verbose=2)
print("Testing set Mean Absolute Error: {:5.2f} MPG".format(mae))

test_predictions = model.predict(normalized_test).flatten()

plt.figure()
plt.scatter(test_labels, test_predictions)
plt.xlabel("True Values [MPG]")
plt.ylabel("Predictions [MPG]")
plt.axis("equal")
plt.axis("square")
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])

plt.figure()
error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")
plt.show()
