import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks

import matplotlib.pyplot as plt
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')

# Setup train and valid sets
train = pd.read_csv("/kaggle/input/asl-dataset-landmarked/train_landmarks.csv")
valid = pd.read_csv("/kaggle/input/asl-dataset-landmarked/valid_landmarks.csv")

X_train = train.copy()
X_valid = valid.copy()
y_train = X_train.pop("label")
y_valid = X_valid.pop("label")

# No preprocessing, the data inside the csv does not contain any empty values or values outside of [0, 1)

print(X_train.shape)
print(y_train.shape)
print(X_valid.shape)
print(y_valid.shape)

input_shape = [X_train.shape[1]]

# Debugging
print(y_train.shape, y_train.dtype)
print(y_valid.shape, y_valid.dtype)

print("Unique train labels:", np.unique(y_train)[:27])
print("Min label:", y_train.min(), "Max label:", y_train.max())

print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)


# =========================
# TRAINING
# =========================

early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    min_delta=1e-4,
    restore_best_weights=True
)

model = keras.Sequential([
    layers.Input(shape=(42,)),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    # layers.Dropout(0.1),
    layers.Dense(64, activation='relu'),
    # layers.Dropout(0.1),
    layers.Dense(27, activation="softmax"),
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=["accuracy"]
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=150,
    callbacks=[early_stop],
)

# View training graph

history_df = pd.DataFrame(history.history)
history_df.loc[5:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()))

# Save model (done on kaggle)
model.save("/kaggle/working/asl_landmark_model.keras")