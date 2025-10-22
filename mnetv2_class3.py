# import the necessary packages
# from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
# import matplotlib.pyplot as plt
import imutils
import cv2
import os
# import copy
# import pickle

print()
# load the trained MobileNetV2
print("[INFO] loading MobileNetV2...")
saved_model = load_model("trainedMNetV2.h5")
print()

# load the input image from disk, clone it, and grab the image spatial
# dimensions
print("[INFO] loading image...")
Image1 = cv2.imread("C:/Users/nomya/Desktop/MScProjExec/9.MNetV2Proj3/ClassExamples/1_RandCMImage1.jpg")
original = Image1.copy()
(h, w) = Image1.shape[:2]
print()

# pre-process the image for classification
# construct a blob from the image
# blob methods used to prepare input images for classification
# via pre-trained deep learning models
# blob methods perform mean subtraction and scaling
# Not required for all DL networks for
# pre-processing of images
print("[INFO] preprocessing the input image...")
Image1 = cv2.resize(Image1, (224, 224))
blob = cv2.dnn.blobFromImage(Image1, 1.0, (300, 300),
	(104.0, 177.0, 123.0))
Image1 = cv2.resize(Image1, (224, 224))
Image1 = img_to_array(Image1)
Image1 = preprocess_input(Image1)
Image1 = np.expand_dims(Image1, axis=0)
print()

# classify the input image
print("[INFO] classifying the input image...")
(Authentic, Forged) = saved_model.predict(Image1)[0]

# build the label
label = "Authentic Image" if Authentic > Forged else "Forged Image"
color = (0, 255, 0) if label == "Authentic Image" else (0, 0, 255)

# include the probability in the label
label = "{}: {:.2f}%".format(label, max(Authentic, Forged) * 100)

# draw the label on the image
output = imutils.resize(original, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)

# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)
