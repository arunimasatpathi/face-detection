import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np
import ctypes

haar_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Train for without mask
capture = cv2.VideoCapture(0)
data_withoutmask = []
while True:
    flag, img = capture.read()
    if flag:
        faces = haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
            face = img[y:y+h, x:x+w, :]
            face = cv2.resize(face, (50,50))
            print(len(data_withoutmask))
            if len(data_withoutmask) < 300:
                data_withoutmask.append(face)
        cv2.imshow('result', img)
        if cv2.waitKey(2) == 27 or len(data_withoutmask) == 300:
            break
capture.release()
cv2.destroyAllWindows()

# Prompt to wear a mask
def Mbox(title, text, style):
    return ctypes.windll.user32.MessageBoxW(0, text, title, style)
Mbox('Alert!', 'Please wear mask!', 1)

# Train for with mask
capture = cv2.VideoCapture(0)
data_withmask = []
while True:
    flag, img = capture.read()
    if flag:
        faces = haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
            face = img[y:y+h, x:x+w, :]
            face = cv2.resize(face, (50,50))
            print(len(data_withmask))
            if len(data_withmask) < 300:
                data_withmask.append(face)
        cv2.imshow('result', img)
        if cv2.waitKey(2) == 27 or len(data_withmask) == 300:
            break
capture.release()
cv2.destroyAllWindows()

# Save the data in numpy file
np.save('without_mask.npy', data_withoutmask)
np.save('with_mask.npy', data_withmask)

