# Import the necessary packages
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
#from keras import backend as K


class SmallerVGGNet:
	def build(width, height, depth, classes):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		svggnmodel = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1
        
        # CONV => RELU => POOL
		svggnmodel.add(Conv2D(32, (3, 3), padding="same",
			input_shape=inputShape))
		svggnmodel.add(Activation("relu"))
		svggnmodel.add(BatchNormalization(axis=chanDim))
		svggnmodel.add(MaxPooling2D(pool_size=(3, 3)))
		svggnmodel.add(Dropout(0.25))
        
        # (CONV => RELU) * 2 => POOL
		svggnmodel.add(Conv2D(64, (3, 3), padding="same"))
		svggnmodel.add(Activation("relu"))
		svggnmodel.add(BatchNormalization(axis=chanDim))
		svggnmodel.add(Conv2D(64, (3, 3), padding="same"))
		svggnmodel.add(Activation("relu"))
		svggnmodel.add(BatchNormalization(axis=chanDim))
		svggnmodel.add(MaxPooling2D(pool_size=(2, 2)))
		svggnmodel.add(Dropout(0.25))
        
        # (CONV => RELU) * 2 => POOL
		svggnmodel.add(Conv2D(128, (3, 3), padding="same"))
		svggnmodel.add(Activation("relu"))
		svggnmodel.add(BatchNormalization(axis=chanDim))
		svggnmodel.add(Conv2D(128, (3, 3), padding="same"))
		svggnmodel.add(Activation("relu"))
		svggnmodel.add(BatchNormalization(axis=chanDim))
		svggnmodel.add(MaxPooling2D(pool_size=(2, 2)))
		svggnmodel.add(Dropout(0.25))
        
        # first (and only) set of FC => RELU layers
		svggnmodel.add(Flatten())
		svggnmodel.add(Dense(512))
		svggnmodel.add(Activation("relu"))
		svggnmodel.add(BatchNormalization())
		svggnmodel.add(Dropout(0.5))

		# softmax classifier
		svggnmodel.add(Dense(classes))
		svggnmodel.add(Activation("softmax"))

		# return the constructed network architecture
		return svggnmodel
        