import cv2 as cv
import mediapipe as mp
from tensorflow import keras

# hands landmark detecting
mp_hands = mp.solutions.hands
# utils for drawing landmark positions on image
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# initialize hands landmark detecting
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

model = keras.models.load_model("asl_landmark_model.keras")

# Hard code in the label map
LABEL_MAP = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", 
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z"
] 

# Open webcam
webcam = cv.VideoCapture(0)

print("main - Webcam started")

# frame by frame
while True:
    ret, frame = webcam.read()

    # Mediapipe requires RGB, convert to such
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    landmark_coords = []

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Check for full set of landmarks before adding to feature
            if len(landmarks.landmark) < 21:
                continue
            for i, lm in enumerate(landmarks.landmark):
                row[f"x{i}"] = lm.x
                row[f"y{i}"] = lm.y
        print(f"LANDMARKS: {landmarks.landmark}")
        print(f"LANDMARK_COORDS: {landmark_coords}")
        if len(landmark_coords) > 0:
            # Make guess
            prediction = model.predict(X=[landmark_coords])
            # with probability weight
            probabilities = model.predict_proba([landmark_coords])

            # print(LABEL_MAP[prediction[0]])
            # print(probabilities[0])
            # print()

            # Only label guesses with >0.7 probability
            # if (probabilities[0][prediction[0]] > 0.7):
            print(LABEL_MAP[prediction[0]], probabilities[0][prediction[0]])
            # else:
                # print("NONE")

    cv.imshow("Webcam", frame)
    if cv.waitKey(40) & 0xFF == 27: # ESC key to exit
        break


webcam.release()
cv.destroyAllWindows()