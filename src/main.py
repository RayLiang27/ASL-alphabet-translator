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

letter_duration = 2.5   # time in between letters
letter_grace_duration = 1.5 # time for person to switch to next letter
was_last_none = False    # was the last letter none (2 nones in a row = clear list)

# frame by frame
while True:
    ret, frame = webcam.read()
    now = time.time()

    # Mediapipe requires RGB, convert to such
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    new_letter = "NONE"

    ### Run thru model
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
            

        npy_row = np.array(row)

        if len(row) > 0:
            guess = model.predict(npy_row, verbose=0)
            # Make guess
            prediction = np.argmax(guess)
            probability = np.max(guess)

            # Only label guesses with great enough certainty
            if (probability > 0.65):
                print(LABEL_MAP[prediction], probability)
                new_letter = LABEL_MAP[prediction]
            else:
                print("NONE")
                new_letter = "NONE"

            # get frame dimensions to convert normalized coords -> pixel coords
            h, w, _ = frame.shape

            x_min = int(min(row_x) * w)
            x_max = int(max(row_x) * w)
            y_min = int(min(row_y) * h)
            y_max = int(max(row_y) * h)

            # optional: add a little padding around the hand
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            # draw bbox around hand + prediction
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 0), 3)
            cv2.putText(frame, new_letter   , (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    ### Determine most likely letter desired by user
    if (now - start_time >= letter_grace_duration):
        if last_letter is None:
            letter_time["NONE"] += now - last_time
        else:
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
        
    ### Write text list onto the screen
    text = "".join(letter_list)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.1
    thickness = 3
    margin = 15  # left/right padding
    h, w, _ = frame.shape
    max_width = w - 2 * margin

    # Trim from the left (oldest letters) until the text fits on screen
    display_text = text
    (text_w, text_h), _ = cv2.getTextSize(display_text, font, font_scale, thickness)
    while text_w > max_width and len(display_text) > 0:
        display_text = display_text[1:]
        (text_w, text_h), _ = cv2.getTextSize(display_text, font, font_scale, thickness)

    text_x = margin
    text_y = h - 30  # near bottom of frame
    box_top = text_y - text_h - 15

    # Background box behind text for readability
    cv2.rectangle(frame, (0, box_top), (w, h), (0, 0, 0), -1)

    ### Indicator for when each letter will tick over
    progress = min((now - start_time) / letter_duration, 1.0)
    circle_radius = 10
    circle_center = (margin + circle_radius, box_top - circle_radius - 10)

    # Background ring (empty state)
    cv2.circle(frame, circle_center, circle_radius, (80, 80, 80), 2)

    # Filled pie slice growing clockwise from the top as time elapses
    end_angle = 360 * progress
    cv2.ellipse(frame, circle_center, (circle_radius, circle_radius),
                -90, 0, end_angle, (0, 255, 0), -1)

    cv2.putText(frame, display_text, (text_x, text_y),
                font, font_scale, (0, 255, 0), thickness)


    cv2.imshow("Webcam", frame)
    if cv2.waitKey(40) & 0xFF == 27: # ESC key to exit
        break


webcam.release()
cv2.destroyAllWindows()