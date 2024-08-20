import os
import cv2

cap = cv2.VideoCapture(0)
dataDir = './data'
if not os.path.exists(dataDir):
    os.makedirs(dataDir)

numberOfLetters = 4
dataSize = 100

for i in range(1, numberOfLetters):
    if not os.path.exists(os.path.join(dataDir, str(i))):
        os.makedirs(os.path.join(dataDir, chr(i+96)))
    print('Press q to collect data for letter {}'.format(chr(i+96)))

    while True:
        success, img = cap.read()
        cv2.putText(img, "To collect data press q", (100, 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 3)
        cv2.imshow("Data_collection", img)
        if cv2.waitKey(25) == ord("q"):
            break

    counter = 0
    while counter < dataSize:
        success, img = cap.read()
        cv2.imshow("Data_collection", img)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(dataDir, chr(i+96), '{}.jpg'.format(counter)), img)
        counter += 1
    print("done for class {}".format(chr(i+96)))

cap.release()
cv2.destroyAllWindows()
