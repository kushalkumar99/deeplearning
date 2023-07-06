#Object recognistion using pre-trained model such as MobileNet-SSD
#SSD-Single Shot Multibox Dectector
#DNN-deep neural network
#with this DNN we are to load the pre trained model into computer vision
# for i in range(1,10,2):
#     print(i)

import numpy as np
import imutils
import cv2
import time

prototxt = "MobileNetSSD_deploy.prototxt.txt"
model = "MobileNetSSD_deploy.caffemodel"
confThresh = 0.2

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor","pen"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("Loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)
print("Model Loaded")
print("Starting Camera Feed...")
vs = cv2.VideoCapture(0)
time.sleep(2.0)

while True:
	_,frame = vs.read()
	frame = imutils.resize(frame, width=500)

	(h, w) = frame.shape[:2]# ploting rectangle aroung=d objects
	imResizeBlob = cv2.resize(frame, (300, 300))
	blob = cv2.dnn.blobFromImage(imResizeBlob,
		0.007843, (300, 300), 127.5)

	net.setInput(blob)
	detections = net.forward()
	detShape = detections.shape[2]
	for i in np.arange(0,detShape): #from 0 to 100 it will process
		confidence = detections[0, 0, i, 2]
		if confidence > confThresh:     
			idx = int(detections[0, 0, i, 1]) #index value [0,1,2,...,22]
			# print("ClassID:",detections[0, 0, i, 1])
			# if idx == 5.0:
			# 	print("i need water, more water")
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h]) #3:7-> 4 vlaues to draw box or rect
			(startX, startY, endX, endY) = box.astype("int")
			
			label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			if startY - 15 > 15:
				y = startY - 15
			else:
				y = startY + 15
			cv2.putText(frame, label, (startX, startY),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
			
			if CLASSES[idx]=="bottle":
				cv2.putText(frame,"I NEED WATER",(startX,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
			

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1)
	if key == 27:
		break
vs.release()
cv2.destroyAllWindows()


    