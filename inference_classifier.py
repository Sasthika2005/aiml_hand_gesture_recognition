
import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
from difflib import get_close_matches

# Load trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize camera
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# ASL Labels Dictionary
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

# Common words dictionary for auto-correction
common_words = ["what", "name", "you", "hello", "how", "are", "thank", "please", "yes", "no"]

def autocorrect(word):
    matches = get_close_matches(word, common_words, n=1, cutoff=0.7)
    return matches[0] if matches else word

last_prediction = None
last_time = time.time()
display_time = 2.0  # Delay between predictions
predicted_word = ""
hand_present = False
corrected_word = ""

while True:
    data_aux = []
    x_ = []
    y_ = []
    ret, frame = cap.read()
    if not ret:
        continue

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_present = True
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

        # Ensure the input size matches what the model expects (84 features)
        if len(data_aux) < 84:
            data_aux.extend([0] * (84 - len(data_aux)))
        data_aux = np.asarray(data_aux).reshape(1, -1)

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10

        current_time = time.time()
        if current_time - last_time > display_time:
            try:
                prediction = model.predict(data_aux)
                last_prediction = labels_dict[int(prediction[0])]
                last_time = current_time
                predicted_word += last_prediction
                corrected_word = autocorrect(predicted_word.strip())
                print(f"Predicted Word: {corrected_word}")
            except Exception as e:
                print(f"Prediction error: {e}")

    else:
        if hand_present:
            predicted_word += " "
            hand_present = False
            corrected_word = autocorrect(predicted_word.strip())
            print(f"Predicted Word: {corrected_word}")

    if last_prediction and 'x1' in locals():
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 3)
        cv2.putText(frame, last_prediction, (x1, y1 - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.putText(frame, corrected_word, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('ASL Sign Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# Initialize camera
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# ASL Labels Dictionary
labels_dict = {
   0: 'A', 1: 'B', 2: 'C',3: 'D',4: 'E',5: 'F',6: 'G',7: 'H',8: 'I',9: 'J',10:'K',11:'L',12: "M",13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'T',21:'U',22:'V',23:'W',24:'X',25:'Y',26:'Z'}

# Common words dictionary for auto-correction
common_words = ["what", "name", "you", "hello", "how", "are", "thank", "please", "yes", "no"]

def autocorrect(word):
    matches = get_close_matches(word, common_words, n=1, cutoff=0.7)
    return matches[0] if matches else word

last_prediction = None
last_time = time.time()
display_time = 2.0  # Increased delay in seconds between predictions
predicted_word = ""
hand_present = False
corrected_word = ""  # Ensure corrected_word is always initialized

while True:
    data_aux = []
    x_ = []
    y_ = []
    ret, frame = cap.read()
    if not ret:
        continue

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        hand_present = True
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

        if len(data_aux) < 84:
            data_aux.extend([0] * (84 - len(data_aux)))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10

        data_aux = np.asarray(data_aux).reshape(1, -1)

        current_time = time.time()
        if current_time - last_time > display_time:
            try:
                prediction = model.predict(data_aux)
                last_prediction = labels_dict[int(prediction[0])]
                last_time = current_time
                predicted_word += last_prediction
                corrected_word = autocorrect(predicted_word)
                print(f"Predicted Word: {corrected_word}")
            except Exception as e:
                print(f"Prediction error: {e}")
    else:
        if hand_present:
            predicted_word += " "  # Add space when hand is removed
            hand_present = False
            corrected_word = autocorrect(predicted_word)
            print(f"Predicted Word: {corrected_word}")

    if last_prediction:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 3)
        cv2.putText(frame, last_prediction, (x1, y1 - 18), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
    
    cv2.putText(frame, corrected_word, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3, cv2.LINE_AA)
    cv2.imshow('ASL Sign Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
