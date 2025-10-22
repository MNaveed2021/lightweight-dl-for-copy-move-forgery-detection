# import the necessary packages
#import matplotlib
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
#from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import plot_confusion_matrix
from svggn4projmodule.svggn4_main import SmallerVGGNet
from imutils import paths
#import seaborn as sns
import numpy as np
#import argparse
import random
#import pickle
import cv2
import os

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 100
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (96, 96, 3)

# initialize the data and labels
data = []
labels = []

# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images("AuFImgDataset")))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
	image = img_to_array(image)
	data.append(image)

	# extract the class label from the image path
    # update the abels list
	label = imagePath.split(os.path.sep)[-2]
	#label = 1 if label == "Forged" else 0
	labels.append(label)
        
# convert the data and labels to NumPy arrays
# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] data matrix: {:.2f}MB".format(
	data.nbytes / (1024 * 1000.0)))

# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# partition the data into training and validation splits using 80% of
# the data for training and the remaining 20% for validation
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.2, stratify=labels, random_state=42)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# initialise the model
print("[INFO] compiling SVGGNet...")
modelsvggn = SmallerVGGNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
	depth=IMAGE_DIMS[2], classes=2)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
modelsvggn.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])
print()

# summarise the model parameters
print("[Summary] SVGGNet...")
print()
modelsvggn.summary()
print()

# train the network
print("[INFO] training SVGGNet...")
H = modelsvggn.fit(
	x=aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)
print()

# make predictions on the testing set
print("[INFO] evaluating SVGGNet...")
print()
print("Classification Report-SVGGNet:")
predIdxs = modelsvggn.predict(testX, batch_size=BS)

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
print("Evaluation Metrics Summary-SVGGNet:")
print()
cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

# show the confusion matrix, accuracy, sensitivity, and specificity
print(cm)
print()
print("Accuracy: {:.4f}".format(acc))
print("Sensitivity: {:.4f}".format(sensitivity))
print("Specificity: {:.4f}".format(specificity))
print()

# plot confusion matrix
# option 1
plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
classNames = ['Authentic','Forged']
plt.colorbar()
plt.title("Confusion Matrix Plot-SVGGNet")
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
plt.title("Training Loss and Accuracy-SVGGNet")
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
plt.title("Validation Loss and Accuracy-SVGGNet")
plt.xlabel("Epochs")
plt.ylabel("Loss / Accuracy")
plt.legend(loc="upper left")
plt.show()

# plot both the training and validation loss & accuracy plots together
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training and Validation Loss / Accuracy-SVGGNet")
plt.xlabel("Epochs")
plt.ylabel("Loss / Accuracy")
plt.legend(loc="upper left")
plt.show()

# To save the figures / plots, set the matplotlib backend to a non-GUI
# using following import & save commands 
# matplotlib.use("Agg")
# plt.savefig("Train&Val Plot")

# save the model to disk
print("[INFO] serializing SVGGNet...")
modelsvggn.save("trainedSVGGNet.h5")
print()
