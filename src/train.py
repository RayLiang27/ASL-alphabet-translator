import os
import cv2 as cv
import mediapipe as mp
import numpy as np

# Scikit-learn
from sklearn.model_selection import train_test_split

raw_data_dir = "../data/raw/" # For the small self-recorded dataset from generate_data.py
# raw_data_dir = "../data/kaggle/" # For the eventual larger data set from Kaggle


loaded = np.load("../data/processed/dataset_small.npz")
# loaded = np.load("../data/processed/dataset_kaggle.npz")
features = loaded["features"]
labels = loaded["labels"]


X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2, shuffle=True, stratify=labels)

# Define model