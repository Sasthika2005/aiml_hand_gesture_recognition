import pickle
import cv2
import mediapipe as mp
import numpy as np
import ollama
import time

# ✅ Load the trained model
model_dict = pickle.load(open("./model.p", "rb"))
model = model_dict["model"]

# ✅ Initialize webcam
cap = cv2.VideoCapture(0)

# ✅ Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)

# ✅ Label dictionary
labels_dict = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I",
    9: "J", 10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q",
    17: "R", 18: "S", 19: "T", 20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z"
}

word = ""
prev_letter = None
last_sign_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_data = []

        # Collect data from up to 2 hands
        for hand_landmarks in results.multi_hand_landmarks[:2]:
            x_ = []
            y_ = []
            single_hand = []

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                single_hand.append(x - min(x_))
                single_hand.append(y - min(y_))

            hand_data.append(single_hand)

            # Draw the hand
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        # Pad with zeros if only one hand detected
        if len(hand_data) == 1:
            hand_data.append([0.0] * 42)

        # Merge both hands’ data into a single array of 84 features
        data_aux = hand_data[0] + hand_data[1]

        # Predict the letter
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        # Draw bounding box around first hand only
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 3)
        cv2.putText(frame, predicted_character, (x1, y1 - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        # Add to word only if different and after a delay
        if predicted_character != prev_letter and (time.time() - last_sign_time) > 1:
            word += predicted_character
            prev_letter = predicted_character
            last_sign_time = time.time()

    # Display accumulated word
    cv2.putText(frame, f"Word: {word}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

    cv2.imshow("Sign Language to AI Chatbot", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# ✅ After loop: Ask chatbot
def extract_keywords(text):
    return text.split() if text else ["Unknown"]

def get_llama_response(text):
    keywords = extract_keywords(text)
    prompt = f"Provide a concise, helpful answer about: {' '.join(keywords)}"

    response = ollama.chat(
        model="llama3.2:latest",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"].strip()

if word:
    print(f"\nDetected Word: {word}")
    chatbot_response = get_llama_response(word)
    print(f"Chatbot Response: {chatbot_response}")
