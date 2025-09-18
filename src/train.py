# Libs
import cv2 as cv
import mediapipe as mp
import numpy as np
import pickle

# Scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

raw_data_dir = "../data/raw/" # For the small self-recorded dataset from generate_data.py
# raw_data_dir = "../data/kaggle/" # For the eventual larger data set from Kaggle


loaded = np.load("../data/processed/dataset_small.npz")
# loaded = np.load("../data/processed/dataset_kaggle.npz")
features = loaded["features"]
labels = loaded["labels"]

print("TRAINING START")

# Test will be done on 2 of the 50 images for each class
X_train, X_test, Y_train, Y_true = train_test_split(features, labels, test_size=0.04, shuffle=True, stratify=labels)

# Define model
model = RandomForestClassifier()
model.fit(X_train, Y_train)

prediction = model.predict(X=X_test)

# Get a quick accuracy score of the model
result = accuracy_score(y_pred=prediction, y_true=Y_true)

print("DONE")

# We should expect 100 as the small database set are all so similar
print(f"Score: {result}")

# Save model with pickle
with open("../models/rand_forest_small.pkl", "wb") as f:
    pickle.dump(model, f)