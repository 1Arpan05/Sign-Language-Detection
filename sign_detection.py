import cv2
import pickle
import numpy as np
import mediapipe as mp

model_dict = pickle.load(open("model.pickle", "rb"))
model = model_dict["model"]

cap = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

data = []

while True:
    data_aux = []
    x_ = []
    y_ = []
    success, img = cap.read()
    h, w, c = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))
        try:
            y_pred = model.predict([np.asarray(data_aux)])
            pred_chr = y_pred[0]
        except ValueError:
            pred_chr = "Detecting two hands"
            continue
        x1 = int(min(x_) * w)
        y1 = int(min(y_) * h)
        x2 = int(max(x_) * w)
        y2 = int(max(y_) * h)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255 , 255), 3)
        cv2.putText(img, pred_chr, (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 3)
    cv2.imshow("Hands", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
