import numpy as np
import imutils
import pickle
import time
import cv2
#DEEP LEARNING- PYTORCH trained model for facial features extraction with the help of embeddings
embeddingModel = "output/openface_nn4.small2.v1.t7"
#embeddingfile for dataset
embeddingFile = "output/embeddings.pickle"
#recognizer is for trained model
recognizerFile = "output/recognizer.pickle"
#lable encoder
labelEncFile = "output/le.pickle"
conf = 0.5
#for face detection
print("Loading face detector...")
prototxt = "model/deploy.prototxt"
#DEEP LEARNING-CAFFE model for face detection
model = "model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt, model)
#for face recognizing using dnn from pytorch
print("Loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(embeddingModel)

recognizer = pickle.loads(open(recognizerFile, "rb").read())#read binary
le = pickle.loads(open(labelEncFile, "rb").read())#read binary

#caffe->face detect->coordinates of face->convert into blobimg->to pass input to the face embedd
#150 imgs->convert into 150 embeddings->append into single file(embeddings.pickle)
#training done with embedding file and saved into two files(recognizer.pickle & le.pickle)

box = []
print("Starting video stream...")
cam = cv2.VideoCapture(0)
time.sleep(2.0)

while True:
    _, frame = cam.read()
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False)

    detector.setInput(imageBlob)
    detections = detector.forward()

    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > conf:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            if fW < 20 or fH < 20:
                continue

            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            preds = recognizer.predict_proba(vec)[0]
            i = np.argmax(preds)
            proba = preds[i]
            name = le.classes_[i]
            text = "{}  : {:.2f}%".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()