from django.shortcuts import render,redirect
from django.shortcuts import get_object_or_404
from django.core.files.storage import FileSystemStorage

import numpy as np
import os
from keras.preprocessing import image
from keras.models import load_model
from  tensorflow.compat.v1 import Session
from tensorflow import Graph

model_graph = Graph()

with model_graph.as_default():
	tf_session = Session()
	with tf_session.as_default():


		# model = load_model("classify/first_one.h5")
		# Please use the below link to download model from drive because it is of large size so
		# i cannot use it here. 
		model = "https://drive.google.com/file/d/1uoa7_WWQXaahpnAB6JbqNZV6Kr9RKm2Z/view?usp=sharing"

IMG_WIDTH = 224
IMG_HEIGHT = 224

labels = {0: "Corn Gray leaf spot", 
1: "Common rust Corn Maize", 
2: "Northern Leaf Blight Corn Maize", 
3: "Healthy Maize Corn", 
4: "Early blight Potato leaf", 
5: "Late blight Potato Leaf", 
6: "healthy Potato Leaf", 
7: "Bacterial spot of Tomato", 
8: "Leaf Mold of Tomato", 
9: "Yellow Leaf Curl Virus Tomato", 
10: "Mosaic virus leaf Tomato", 
11: "Healthy Tomato Leaf"}

print(labels[0])


def index(request):


	return render(request,"index.html")

def predictImage(request):
	print(request.POST.dict())
	fileObj = request.FILES["document"]
	fs=FileSystemStorage()
	filePathName = fs.save(fileObj.name,fileObj)
	filePathName = fs.url(filePathName)

	test_image = "."+filePathName

	img = image.load_img(test_image,target_size=(IMG_WIDTH,IMG_HEIGHT,3))
	img = image.img_to_array(img)
	img = img/255
	x = img.reshape(1,IMG_WIDTH,IMG_HEIGHT,3)
	with model_graph.as_default():
		with tf_session.as_default():
			proba = model.predict(x)
			top_3 = np.argsort(proba[0])[:-4:-1]
	max_proba = np.max(proba)
	if max_proba>0.5:
		for i in range(1):
			label_pred = labels[(top_3[i])]
			acc_pred = proba[0][top_3[i]]*100
	else:
		message = "This image is not of leaf please try other one."
	print(np.max(proba))
	print(label_pred)
	print(acc_pred)
	# prediction = labels[str(np.argmax(proba[0]))]



	context={
	"filePathName":filePathName,
	"prediction":label_pred,
	"Accuracy" : acc_pred,
	}
	return render(request,"test.html",context)

