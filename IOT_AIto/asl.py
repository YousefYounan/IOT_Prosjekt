import cv2
import numpy as np
import time
import mediapipe as mp
import pyttsx3
import threading
from keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D as KerasDepthwiseConv2D


class CustomDepthwiseConv2D(KerasDepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(*args, **kwargs)


model = load_model("keras_Model.h5", custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D}, compile=False)
class_names = [line.strip() for line in open("labels.txt", "r").readlines()]
camera = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

engine = pyttsx3.init()
engine.setProperty('rate', 150)

last_prediction_time = 0
debounce_seconds = 1.0
sentence = ""
last_letter_spoken = ""


def speak(letter):
    engine.say(letter)
    engine.runAndWait()

def preprocess_frame(frame_rgb):
    resized = cv2.resize(frame_rgb, (224, 224)).astype(np.float32)
    normalized = (resized / 127.5) - 1.0
    return np.expand_dims(normalized, axis=0)

def update_sentence(label, confidence):
    global sentence, last_letter_spoken
    clean_label = ''.join(filter(str.isalpha, label))

    if label.lower() == "Mellomrom":
        sentence += " "
    elif label.lower() == "Slett":
        sentence = ""
    else:
        sentence += clean_label
        if clean_label != last_letter_spoken:
            threading.Thread(target=speak, args=(clean_label,), daemon=True).start()
            last_letter_spoken = clean_label

while True:
    ret, frame = camera.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    label = ""
    confidence = 0.0

    if results.multi_hand_landmarks:
        input_data = preprocess_frame(frame_rgb)

        if time.time() - last_prediction_time > debounce_seconds:
            prediction = model.predict(input_data, verbose=0)[0]
            max_index = np.argmax(prediction)
            confidence = prediction[max_index]
            label = class_names[max_index]

            if confidence > 0.80:
                update_sentence(label, confidence)
                last_prediction_time = time.time()

        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    key = cv2.waitKey(1)
    if key == 27:  
        break
    elif key == 8:  
        sentence = ""
    elif key == 32:  
        sentence += " "

    confidence_percent = int(confidence * 100) if label else 0
    display_label = ''.join(filter(str.isalpha, label))

    cv2.rectangle(frame, (5, 5), (400, 40), (50, 50, 50), -1)
    cv2.putText(frame, f"Prediksjon: {display_label.upper()} ({confidence_percent}%)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.rectangle(frame, (5, 60), (635, 100), (0, 0, 0), -1)
    cv2.putText(frame, sentence[-50:], (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("ASL Oversetter", frame)

camera.release()
cv2.destroyAllWindows()
