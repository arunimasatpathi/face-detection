import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np
import ctypes
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Load image
# imagePath = sys.argv[1]

# img = cv2.imread(imagePath)
# print(img.shape)

# print(img[0])

# print(img)

# #plots graph hence losses color
# print(plt.imshow(img))
# while True:
#     cv2.imshow("result", img)
#     if cv2.waitKey(2) == 27:
#         break
# cv2.destroyAllWindows()

# #viola jones- haar features
# #haar cascade data - google -github-xml file- for face detection- haarcascade frontalface default.xml

haar_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# haar_data.detectMultiScale(img)

# cv2.rectangle(img,(x,y),(w,h),(b,g,r),border_thickness)

# while True:
#     faces = haar_data.detectMultiScale(img)
#     for x,y,w,h in faces:
#         cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
#         cv2.imshow('result', img)
#     if cv2.waitKey(2) == 27:
#         break
# cv2.destroyAllWindows()

# capture = cv2.videocapture(0) #to start camera and capture video 0=default camera 1= other camera or pass any path of a video if u want to
# while True:
#     flag, img = capture.read()
#     if flag:
#         faces = haar_data.detectMultiScale(img)
#         for x,y,w,h in faces:
#             cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
# cv2.imshow('result', img)
# if cv2.waitkey(2) == 27:
# break
# capture.release()
# cv2.destroyAllWindows()






# #to save the data in a numpy file
# np.save('with_mask.npy',data)

# #to watch data
# print(plt.imshow(data[0]))


# #to process
# import numpy as np
# import cv2


# x_test = pca.transform(x_test)
# y_pred = svm.predict(x_test)

# print(accuracy_score(y_test, y_pred))




