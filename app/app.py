from flask import Flask, render_template,request, send_file
import numpy as np
import keras.models
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import re
import sys
import os
from scipy import ndimage, misc
from imageio import *
from skimage import *
import base64
from keras.models import model_from_json, load_model
import boto3
import json
import urllib2
from PIL import Image
from io import BytesIO
import tempfile
#from load import * 
#initalize our flask app
client = boto3.client("lambda", region_name=os.environ['AWS_DEFAULT_REGION'], aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY_ID'])

s3client = boto3.client("s3", region_name=os.environ['AWS_DEFAULT_REGION'], aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY_ID'])

app = Flask(__name__)

#Load Model
def loadModel():
	req = urllib2.urlopen('https://s3-us-west-1.amazonaws.com/www.colorimages.com/model/model.json')
	loaded_model_json = req.read()
	loaded_model = model_from_json(loaded_model_json)
	result = s3client.download_file("www.colorimages.com", "model/model.h5", "/tmp/model.h5")
	loaded_model.load_weights("/tmp/model.h5")
	print("Loaded model from disk")
	return loaded_model


#decoding an image from base64 into raw representation
def convertImage(imgData1):
	imgstr = re.search(r'base64,(.*)',imgData1).group(1)
	#print(imgstr)
	imgstr.decode('base64')
	client.invoke(FunctionName="uploadImageToS3", Payload=json.dumps({"imgData": imgstr, "fileName": "grayscaleImage.jpeg"}))
	#with open('output.jpeg','wb') as output:
		#output.write(imgstr.decode('base64'))

@app.route('/')
def index():
	#initModel()
	#render out pre-built HTML file right on the index page
	return render_template("index.html")

@app.route('/predict/',methods=['GET','POST'])
def predict():
	colorize = []
	image = request.get_data()
	convertImage(image)

	req = urllib2.urlopen('https://s3-us-west-1.amazonaws.com/www.colorimages.com/testImages/grayscaleImage.jpeg')
	loadedImage = Image.open(req)
	rgb_img = loadedImage.convert('RGB')
	colorize.append(img_to_array(rgb_img))

	#colorize.append(img_to_array(load_img("output.jpeg")))

	colorize = np.array(colorize, dtype=float)
	#print(colorize)
	colorize = color.rgb2lab(1.0/255*colorize)[:,:,:,0]
	colorize = colorize.reshape(colorize.shape+(1,))

	loaded_model = loadModel()

	#print(colorize)

	output = loaded_model.predict(colorize)
	output = output * 128

	for i in range(len(output)):
		cur = np.zeros((256, 256, 3))
		cur[:,:,0] = colorize[i][:,:,0]
		cur[:,:,1:] = output[i]
		resImage = color.lab2rgb(cur)

	#print(resImage)
	resImage = (resImage * 255).astype('uint8')
	finalResult = Image.fromarray(resImage, 'RGB')

	buffered = BytesIO()
	finalResult.save(buffered, format="JPEG")
	encodedString = base64.b64encode(buffered.getvalue())

	client.invoke(FunctionName="uploadImageToS3", Payload=json.dumps({"imgData": encodedString, "fileName": "coloredImage.jpeg"}))

	return encodedString
	
	

if __name__ == "__main__":
	#decide what port to run the app in
	port = int(os.environ.get('PORT', 80))
	#run the app locally on the givn port
	app.run(host='0.0.0.0', port=port)
	#optional if we want to run in debugging mode
	#app.run(debug=True)