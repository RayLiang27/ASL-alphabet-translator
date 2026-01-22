import os
import cv2 as cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras

# hands landmark detecting
mp_hands = mp.solutions.hands
# utils for drawing landmark positions on image
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# initialize hands landmark detecting
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

# Load model
model = keras.models.load_model("./models/asl_landmark_model.keras")

# Hard code in the label map
LABEL_MAP = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", 
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z", "SPACE"
] 

# Open webcam
webcam = cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(0)
print("main - Webcam started")

# frame by frame
while True:
    ret, frame = webcam.read()

    # Mediapipe requires RGB, convert to such
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            row = [[]]

            # Check for full set of landmarks before adding to feature
            if len(landmarks.landmark) < 21:
                continue
            for i, lm in enumerate(landmarks.landmark):
                row[0].append(lm.x)
                row[0].append(lm.y)

        npy_row = np.array(row)

        if len(row) > 0:
            guess = model.predict(npy_row, verbose=0)
            # Make guess
            prediction = np.argmax(guess)
            probability = np.max(guess)

            # Only label guesses with >0.7 probability
            if (probability > 0.7):
                print(LABEL_MAP[prediction], probability)
            else:
                print("NONE")

    cv2.imshow("Webcam", frame)
    if cv2.waitKey(40) & 0xFF == 27: # ESC key to exit
        break


webcam.release()
cv2.destroyAllWindows()