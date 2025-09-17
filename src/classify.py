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
					continue
				for i in range(len(landmarks.landmark)):
					landmark_coords.append(landmarks.landmark[i].x)
					landmark_coords.append(landmarks.landmark[i].y)
			
			if len(landmark_coords) > 0:
				features.append(landmark_coords)
				labels.append(folder_i)


# Save the classified data into a file
np.savez_compressed("../data/processed/dataset_small.npz", features=features, labels=labels) # for smaller dataset
# np.savez_compressed("../data/processed/dataset_kaggle.npz", features=features, labels=labels) # for larger dataset