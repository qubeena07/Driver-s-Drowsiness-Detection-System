import os
import tensorflow
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt 
import numpy as np
from keras.utils.np_utils import to_categorical
import random,shutil
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout,Conv2D,Flatten,Dense, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model

#loading of the data assets by data pre-processing methods
def generator(dir, gen=image.ImageDataGenerator(rescale=1./255), shuffle=False,batch_size=28,target_size=(32,32),class_mode='categorical' ):

    return gen.flow_from_directory(dir,batch_size=batch_size,shuffle=shuffle,color_mode='grayscale',class_mode=class_mode,target_size=target_size)

#BS- batch size, TS- target size
BS= 28
TS=(32,32)

#image generator of train images in the form of batch size
train_batch= generator('data/train',shuffle=False, batch_size=BS,target_size=TS)

#image generator of test images in the form of batch size
valid_batch= generator('data/test',shuffle=False, batch_size=BS,target_size=TS)

#spe steps for epoch
SPE= len(train_batch.classes)//BS

#VS- validation steps
VS = len(valid_batch.classes)//BS
print(SPE,VS)



#Model for CNN
model = Sequential([
    Conv2D(256, kernel_size=(3,3), activation='relu',input_shape=(32,32,1)),
    MaxPooling2D(pool_size=(2,2)),
    
    Conv2D(128,(3,3),activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
#32 convolution filters used each of size 3x3
#again
    Conv2D(164, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),



#flatten since too many dimensions, we only want a classification output
    Flatten(),
#fully connected to get all relevant data
    Dropout(0.5), 
    Dense(64, activation='relu'),
    
#output a softmax to squash the matrix into output probabilities
    Dense(2, activation='softmax'),
])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#shows summary of the whole model in a the table format
model.summary()

#plot of table in flowchart
plot_model(model, to_file='model.png', show_shapes=True)

#to add the input in model
history = model.fit(train_batch, validation_data=valid_batch,epochs=30,steps_per_epoch=SPE ,validation_steps=VS)
model.save('drow_model.h5', overwrite=True)

#for visualization of accuracy and loss in model in the form of graph
import matplotlib.pyplot as plt
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))

plt.plot(epochs, accuracy, "b", label="trainning accuracy")
plt.plot(epochs, val_accuracy, "r", label="validation accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('Accuracy.png')
plt.show()


plt.plot(epochs, loss, "b", label="trainning loss")
plt.plot(epochs, val_loss, "r", label="validation loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Loss.png')
plt.show()