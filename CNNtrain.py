import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from sklearn.model_selection import train_test_split
# import tensorflow as tf

import tensorflow.compat.v1 as tf

import numpy as np
import pandas as pd


#from tensorflow.keras.callbacks import TensorBoard
from tensorflow.compat.v1.keras.models import Sequential
from tensorflow.compat.v1.keras.layers import BatchNormalization
from tensorflow.compat.v1.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.compat.v1.keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from tensorflow.compat.v1.keras.optimizers import Adadelta, Adam
from tensorflow.compat.v1.keras.callbacks import ModelCheckpoint

# from sklearn.preprocessing import MinMaxScaler

#from tensorflow.keras import callbacks
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()


data = np.load(r"D:/CNNtest2022/Data4Area5C/WFBB_Norm_data5C.npy") # 5 channels data,(num,100,100,5), channel 0-4 is DEM,RGB,NIR
label = np.load(r"D:/CNNtest2022/Data4Area5C/WFBB_label.npy")


            
# split data into train and test groups
train_data,test_data,train_label,test_label = train_test_split(data,label,test_size=0.2,random_state=1,stratify=label)


# creat CNN model
def createCNN():
    model = Sequential()
    

    model.add(Convolution2D(128, 3, 3, padding="same", input_shape=(100, 100, 5)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2,2)))
#    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.30))


    model.add(Convolution2D(256, (3,3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2,2)))
#    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.30))
    

    model.add(Convolution2D(512, (5,5), padding="same"))    
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2,2)))
#    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.30))
    

    model.add(Convolution2D(1024, (5,5), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2,2)))
#    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.30))
    
    model.add(Flatten())
    
    model.add(Dense(512, activation = 'relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(2, activation = 'softmax')) # 2 classes
    
    model.summary()
    
    return model



LEARNING_RATE = 0.001
BATCH_SIZE =32
EPOCHS = 100


model = createCNN()


#Adam optimizer
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(lr=LEARNING_RATE),
              metrics=['accuracy'])





OUT_DIR = r"D:/CNNtest2022/OutModel2022"
checkpoint = ModelCheckpoint(os.path.join(OUT_DIR, 'ModelWFBB_DEM_Aer.h5'),  # model filename
                             monitor='val_accuracy', # quantity to monitor
                             verbose=0, # verbosity - 0 or 1
                             save_best_only= True, # The latest best model will not be overwritten
                             mode='auto') # The decision to overwrite model is made 
                                           # automatically depending on the quantity to monitor

model_details = model.fit(train_data, train_label,
                          batch_size = BATCH_SIZE,
                          epochs = EPOCHS, 
                          validation_split=0.25,
                          callbacks=[checkpoint],
                          verbose=1)

                                           

scroe, accuracy = model.evaluate(test_data, test_label, batch_size=BATCH_SIZE)
print('CNN_Test: loss:', scroe, 'accuracy:', accuracy)

# Predict
pred_label = model.predict(test_data)
#print (pred_label.shape, pred_label)
pred_label = np.argmax(pred_label, axis=1)



#### plot confusion matrix
def plot_confusion_matrix(confusion_mat):
    plt.imshow(confusion_mat,interpolation='nearest',cmap=plt.cm.Wistia)
    plt.title('CNN Confusion Matrix',fontsize=20)
    plt.colorbar()
    tick_marks=np.arange(2) # class number
    plt.xticks(tick_marks,tick_marks,fontsize=16)
    plt.yticks(tick_marks,tick_marks,fontsize=16)
    plt.ylabel('True Label',fontsize=16)
    plt.xlabel('Predicted Label',fontsize=16)
    for i in range(len(confusion_mat)):    #row
        for j in range(len(confusion_mat[i])):    #col
            plt.text(j, i, confusion_mat[i][j],fontsize=16) # images number of each part
    plt.show()
    


confusion_matrix = tf.math.confusion_matrix(test_label,pred_label, num_classes=None, dtype=tf.int32, name=None, weights=None)
# sess=tf.compat.v1.Session()
sess=tf.Session()

#with tf.compat.v1.Session(graph=g) as sess:
confusion_matrix = sess.run(confusion_matrix)
plot_confusion_matrix(confusion_matrix)



######################### learning curve

def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    #plt.gca().set_ylim(0, 1)
    plt.show()
    
plot_learning_curves(model_details)

print ('Accuracy:', '{:.4f}'.format(accuracy), 'Batch Size:', BATCH_SIZE,'Learning Rate:', LEARNING_RATE, 'Epochs:', EPOCHS)