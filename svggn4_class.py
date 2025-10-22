# import the necessary packages
# from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
# import matplotlib.pyplot as plt
import imutils
import cv2
# import os
# import copy
# import pickle

print()
# load the trained MNetV2
print("[INFO] loading SVGGNet...")
saved_model = load_model("trainedSVGGNet.h5")
print()

# load the image
# make a shallow copy of the original image
print("[INFO] loading image...")
Image1 = cv2.imread("C:/Users/nomya/Desktop/MScProjExec/8.SVGGNProj4/ClassExamples/sony_61tamp4.jpg")
original1 = Image1.copy()
print()

# pre-process the image for classification
print("[INFO] preprocessing the input image...")
Image1 = cv2.resize(Image1, (96, 96))
Image1 = Image1.astype("float") / 255.0
Image1 = img_to_array(Image1)
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
output = imutils.resize(original1, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)

# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)


## Copied from previous model 
# load the image
# make a shallow copy of the original image
# Image1 = cv2.imread("3001_F_BC2.png")
# original1 = Image1.copy()

# pre-process the image for classification
#Image1 = cv2.resize(Image1, (96, 96))
#Image1 = Image1.astype("float") / 255.0
#Image1 = img_to_array(Image1)
#Image1 = np.expand_dims(Image1, axis=0)
#print()

# load the trained SmallerVGGNet
#print("[INFO] loading SVGGNet...")
#print()
#saved_model = load_model("trainedSVGGN1.h5")
# classify the input image
#print("[INFO] classifying image...")
#(Authentic, Forged) = saved_model.predict(Image1)[0]

# build the label
#label = "Forged Image" if Forged > Authentic else "Authentic Image"
#proba = Forged if Forged > Authentic else Forged
#label = "{}: {:.2f}%".format(label, proba * 100)
# draw the label on the image
#output = imutils.resize(original1, width=400)
#cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	#0.7, (0, 255, 0), 2)

# show the output image
#cv2.imshow("Output", output)
#cv2.waitKey(0)
