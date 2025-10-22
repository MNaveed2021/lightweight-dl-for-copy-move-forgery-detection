# import the necessary packages
#import matplotlib
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.layers import AveragePooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
#import argparse
import os
import random
#import pickle
import cv2

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 100
INIT_LR = 1e-3
BS = 32

# grab the image paths, then
# initialise the data and labels (class images)
print("[INFO] loading images...")
imagePaths = list(paths.list_images("AuFImgDataset"))
data = []
labels = []
print()

# loop over the image paths
for imagePath in imagePaths:
	# extract the class label from the filename
	label = imagePath.split(os.path.sep)[-2]
	
    # load the input image (224x224) and preprocess it
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)
    
    # update the data and labels lists, respectively
	data.append(image)
	labels.append(label)
    
# convert the data and labels to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)
    
# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# partition the data into training and validation splits using 80% of
# the data for training and the remaining 20% for validation
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.2, stratify=labels, random_state=42)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.2,
	height_shift_range=0.2, shear_range=0.15, zoom_range=0.15,
	horizontal_flip=True, fill_mode="nearest")

# load the MobileNetV2 network, ensuring the head FC layer sets are
# left off
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

# initialise the model
print("[INFO] compiling MobileNetV2...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])
print()

# summarise the model parameters
print("[Summary] MobileNetV2...")
print()
model.summary()
print()

# train the network
print("[INFO] training MobileNetV2...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
	epochs=EPOCHS, verbose=1)
print()

# make predictions on the testing set
print("[INFO] evaluating MobileNetV2...")
print()
print("Classification Report-MobileNetV2:")
predIdxs = model.predict(testX, batch_size=BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)
print()

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))
print()

# compute the confusion matrix and and use it to derive the raw
# accuracy, sensitivity, and specificity
print("Evaluation Metrics Summary-MobileNetV2:")
print()
cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

# show the confusion matrix, accuracy, sensitivity, and specificity
print(cm)
print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))
print()

# plot confusion matrix
# option 1
plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
classNames = ["Authentic","Forged"]
plt.colorbar()
plt.title("Confusion Matrix Plot-MobileNetV2")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
thresh = cm.max() / 2
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j]) + " = " + str(cm[i][j]),
        color='white' if cm[i,j] > thresh else 'black')
plt.show()

# option 2 for confusion matrix
#group_names = ["TN","FP","FN","TP"]
#group_counts = ["{0:0.0f}".format(value) for value in
                #cm.flatten()]
#group_percentages = ["{0:.2%}".format(value) for value in
                     #cm.flatten()/np.sum(cm)]
#labels1 = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          #zip(group_names, group_counts, group_percentages)]
#labels1 = np.asarray(labels1).reshape(2,2)
#sns.heatmap(cm, annot=labels1, fmt="", cmap="Blues")

# Alternative option for cm
#sns.heatmap(cm/np.sum(cm), annot=True, 
            #fmt='.2%', cmap="Blues")

# plot the training loss and accuracy plot only
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.title("Training Loss and Accuracy-MobileNetV2")
plt.xlabel("Epochs")
plt.ylabel("Loss / Accuracy")
plt.legend(loc="upper left")
plt.show()

# plot the validation loss and accuracy plot only
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Validation Loss and Accuracy-MobileNetV2")
plt.xlabel("Epochs")
plt.ylabel("Loss / Accuracy")
plt.legend(loc="upper left")
plt.show()

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training and Validation Loss / Accuracy-MobileNetV2")
plt.xlabel("Epochs")
plt.ylabel("Loss / Accuracy")
plt.legend(loc="lower left")
plt.show()

# save the model to disk
print("[INFO] saving MobileNetV2...")
model.save("trainedMNetV2.h5")
print()
