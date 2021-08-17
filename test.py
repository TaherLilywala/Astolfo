from astolphise import Astolphise
import numpy as np
import cv2
import keras.preprocessing.image

a = Astolphise()

SIZE = 160
img = cv2.imread('./non asti.png') 
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # open cv reads images in BGR format so we have to convert it to RGB
img = cv2.resize(img, (SIZE, SIZE)) #resizing image
img = img.astype('float32') / 255.0

test_img = keras.preprocessing.image.img_to_array(img)
test_img = np.array(test_img)

test_img = np.reshape(test_img,(-1,SIZE,SIZE,3))

Prediction = a.predict(test_img)
    
maxEle = max(Prediction[0])
maxIndex = np.where(maxEle == Prediction)

if maxIndex[1][0] == 0:
    print("Yay, It's astolfo  :eggplant:")
else:
    print("sad")

