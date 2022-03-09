# make a prediction for a new image.
from numpy import argmax
from PIL import ImageFile, Image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

# load and prepare the image
def preprocess_image(img_bytes):
	# load the image
	img = Image.open(img_bytes)
	img = img.convert('L')
	img = img.resize((28, 28), Image.NEAREST)
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 1 channel
	img = img.reshape(1, 28, 28, 1)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img
 
# load an image and predict the class
def classifyImage(file):
	# load the image
	img = preprocess_image(file)
	# load model
	model = load_model('/Users/sangameshitagi/Documents/digit-classification/models/digit_class_model.h5')
	# predict the class
	predict_value = model.predict(img)
	digit = argmax(predict_value)
	print(digit)
	return digit