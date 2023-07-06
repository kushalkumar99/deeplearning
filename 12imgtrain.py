#image classification is a supervised learning problem, we need to label the data b4 classification
# using CNN for this classification
# CNN ARCHITECTURE = INPUT->CONVOLUTIONAL LAYER->POOLING->FLATTENING->OUTPUT NEURONS

#FULLY connected later->units=128,activation'relu'
#output layer->units=1,activation="sigmoid"
#sigmoid either gives 1 or 0, 
#compiling =accuracy based modification of weights on our training
#steps_per_epoch = no. of img in training dataset/batch size
#epoch-iteration
#validation_steps = np.img in validation dataset/batch size
#pip install simple_image_download==0.4 this earlier version imagedownloader will help


# from simple_image_download import simple_image_download as sim
# resp = sim.simple_image_download
# resp().download('thor',50)
# code to download imags from google and create a separate dataset 


#code for renaming all imgs into one name
# import os
# os.chdir('')
# i = 1
# for file in os.listdir():
#     src = file
#     dst = 'thor'+str(i)
#     os.remove(src,dst)
#     i+=1


#TRAINING(train) IMAGE CLASSIFICATION USING CNN

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator


#designing CNN
model = Sequential()
model.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))

#compile the model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#preprocessing training arguments
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
#arguments for validation
val_datagen=ImageDataGenerator(rescale=1./255)
#training the model
training_set = train_datagen.flow_from_directory('dataset1/train',
                                                 target_size=(64,64),
                                                 batch_size=8,
                                                 class_mode='binary')
#validation of model
val_set = val_datagen.flow_from_directory('dataset1/val',
                                                 target_size=(64,64),
                                                 batch_size=8,
                                                 class_mode='binary')
#fitting the model
model.fit_generator(training_set,
                    steps_per_epoch=10,
                    epochs=50,
                    validation_data=val_set,
                    validation_steps=2)

model_json=model.to_json()
with open("model,json","w") as json_file:
    json_file.write(model_json)
#saving the model
model.save_weights("model.h5")
print("Saved model to disk")