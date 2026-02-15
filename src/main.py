# basic libs
import os
import time
from collections import defaultdict

# model libs
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

# Letter list
letter_list = []

# Detect which letter is being shown for the most time as the letter to list
start_time = time.time()
letter_time = defaultdict(float)
last_letter = None
last_time = time.time()

letter_duration = 2.5   # time taken to be counted as a full letter
was_last_none = False    # was the last letter none (2 nones in a row = clear list)

# frame by frame
while True:
    ret, frame = webcam.read()
    now = time.time()

    # Mediapipe requires RGB, convert to such
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    new_letter = "NONE"

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            row_x = []
            row_y = []
            row = [[]]

            # Check for full set of landmarks before adding to feature
            if len(landmarks.landmark) < 21:
                continue
            for i, lm in enumerate(landmarks.landmark):
                row[0].append(lm.x)
                row[0].append(lm.y)

                row_x.append(lm.x)
                row_y.append(lm.y)
            
            # draw bbox around hand + prediction
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 3)

        npy_row = np.array(row)

        if len(row) > 0:
            guess = model.predict(npy_row, verbose=0)
            # Make guess
            prediction = np.argmax(guess)
            probability = np.max(guess)

            # Only label guesses with >0.7 probability
            if (probability > 0.7):
                print(LABEL_MAP[prediction], probability)
                new_letter = LABEL_MAP[prediction]
            else:
                print("NONE")
                new_letter = "NONE"

    if last_letter is not None:
        letter_time[last_letter] += now - last_time

    last_letter = new_letter
    last_time = now

    # 2-second window elapsed
    if now - start_time >= letter_duration:
        highest = max(letter_time, key=letter_time.get)

        if highest == "SPACE":
            highest = " "

        if highest == "NONE":
            if was_last_none:
                # clear letter buffer
                print("\t BUFFER CLEARED")
                letter_list = []
                was_last_none = False
            else:
                was_last_none = True
        else:
            letter_list.append(highest)
            was_last_none = False
        
        print(f" === {highest} === ")
        print(letter_list)

        # Reset counters
        letter_time.clear()
        start_time = now



    cv2.imshow("Webcam", frame)
    if cv2.waitKey(40) & 0xFF == 27: # ESC key to exit
        break


webcam.release()
cv2.destroyAllWindows()