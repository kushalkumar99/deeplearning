#SVM for training, here we should use classification
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

#initializing of embedding & recognizer pickle
embeddingFile = "output/embeddings.pickle"
#new and empty at initial 
recognizeFile = "output/recognizer.pickle"
labelEnFile = "output/le.pickle"

print("loading face embeddings")
data = pickle.loads(open(embeddingFile,"rb").read())#read binary

print("encoding labels..")
labelEnc = LabelEncoder()
labels= labelEnc.fit_transform(data["names"]) #inside data we should only get names

print("training model")
recognizer = SVC(C=1.0,kernel="linear",probability=True)#classifier threshold value
recognizer.fit(data["embeddings"],labels)

f = open(recognizeFile,"wb")
f.write(pickle.dumps(recognizer))
f.close()

f = open(labelEnFile,"wb")
f.write(pickle.dumps(labelEnc))
f.close()