import os
import cv2 as cv
import mediapipe as mp
import numpy as np

# hands landmark detecting
mp_hands = mp.solutions.hands

# initialize hands landmark detecting
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


raw_data_dir = "../data/raw/" # For the small self-recorded dataset from generate_data.py
# raw_data_dir = "../data/kaggle/" # For the eventual larger data set from Kaggle

data_folders = os.listdir(raw_data_dir)

features = []
labels = []

missed_data = {
    "A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0, "G": 0, "H": 0, "I": 0, "J": 0,
    "K": 0, "L": 0, "M": 0, "N": 0, "O": 0, "P": 0, "Q": 0, "R": 0, "S": 0, "T": 0,
    "U": 0, "V": 0, "W": 0, "X": 0, "Y": 0, "Z": 0
}

print("=================================")
print("STARTING")
print("=================================")

for folder_i, folder in enumerate(data_folders):
    for file in os.listdir(f"{raw_data_dir}/{folder}"):

        # MediaPipe requires RGB images
        img = cv.imread(f"{raw_data_dir}/{folder}/{file}")
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        landmark_coords = []

        # Run landmark detection
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:

            for landmarks in results.multi_hand_landmarks:
                # Check for full set of landmarks before adding to feature
                if len(landmarks.landmark) < 21:
                    missed_data[folder] += 1
                    continue
                for i in range(len(landmarks.landmark)):
                    landmark_coords.append(landmarks.landmark[i].x)
                    landmark_coords.append(landmarks.landmark[i].y)
            
            if len(landmark_coords) > 0:
                features.append(landmark_coords)
                labels.append(folder_i)

features = np.asarray(features)
labels = np.asarray(labels)

print("")
print("")
print("=================================")
print("RESULTS")
print("=================================")
print(f"features: {len(features)}")
print(features)
print(f"labels: {len(labels)}")
print(labels)

print("")
print("=================================")
print("MISSED")
print("=================================")
has_missed = False
for key in missed_data:
    if missed_data[key] > 0:
        print(f"\t{key}: {missed_data[key]}")
        has_missed = True
if not has_missed:
    print("\tNONE")

# Save the classified data into a file
np.savez_compressed("../data/processed/dataset_small.npz", features=features, labels=labels, label_map=data_folders) # for smaller dataset
# np.savez_compressed("../data/processed/dataset_kaggle.npz", features=features, labels=labels) # for larger dataset