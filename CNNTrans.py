import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from skimage import io
import glob

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

from sklearn.preprocessing import MinMaxScaler

#from tensorflow.keras import callbacks
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()


# path='D:/CNNtest/data/'   #DEM

pathFWBB='D:/CNNtest2022/NE_FWBB_DEM/'   #using DEM from NE FWBB to test existed model
pathIL='D:/CNNtest2022/IL_DEM/'  #using DEM from IL
pathCA='D:/CNNtest2022/CA_DEM/'  #using DEM from CA
pathND='D:/CNNtest2022/ND_DEM/'  #using DEM from ND

# using one watershed to train model, 3 of them to test
pathTrain=pathFWBB
pathTest1=pathCA
pathTest2=pathIL
pathTest3=pathND

# read all images
def read_img(path):
    cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)] # get F,T folder
    imgs=[]
    labels=[]
    imgs_name=[]
    for idx,folder in enumerate(cate): # idx-> 0:F; 1:T; folder-> F,T
        for im in glob.glob(folder+"/*.tif"):
            im = im.replace('\\','/')
#            print('reading the images:%s'%(im))

            img_name=os.path.basename(im)
            img_name=os.path.splitext(img_name)[0] # get file name  
                      
            img=io.imread(im)
            
            # Normalize the dataset-MaxMin
            scaler = MinMaxScaler(feature_range=(0, 1))
            img = scaler.fit_transform(img)
            
            imgs.append(img)
            labels.append(idx)
            imgs_name.append(img_name) # image name
            
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32), np.asarray(imgs_name)

data,label,name=read_img(pathTrain)

#### Test data-3 watersheds
dataTst1, labelTst1, nameTst1 =read_img(pathTest1)
dataTst2, labelTst2, nameTst2 =read_img(pathTest2)
dataTst3, labelTst3, nameTst3 =read_img(pathTest3)



###############################
data=np.expand_dims(data,3) # add channel dimension, 4D


labelN=label.shape[0]


label_name=np.empty((labelN,2),dtype=int) # label + name
for i in range(labelN):
    label_name[i,0]=label[i] # label
    if (name[i].isdigit()):
        name[i]=name[i]
    else:
        name[i]=name[i][1:] # only keep number, delete first letter
    label_name[i,1]=name[i]



############## Test
dataTst1=np.expand_dims(dataTst1,3)
dataTst2=np.expand_dims(dataTst2,3)
dataTst3=np.expand_dims(dataTst3,3)

######Test1
labelN_Tst1=labelTst1.shape[0]

label_nameTst1=np.empty((labelN_Tst1,2),dtype=int) # label + name
for i in range(labelN_Tst1):
    label_nameTst1[i,0]=labelTst1[i] # label
    if (nameTst1[i].isdigit()):
        nameTst1[i]=nameTst1[i]
    else:
        nameTst1[i]=nameTst1[i][1:] # only keep number, delete first letter
    label_nameTst1[i,1]=nameTst1[i]



######Test2
labelN_Tst2=labelTst2.shape[0]

label_nameTst2=np.empty((labelN_Tst2,2),dtype=int) # label + name
for i in range(labelN_Tst2):
    label_nameTst2[i,0]=labelTst2[i] # label
    if (nameTst2[i].isdigit()):
        nameTst2[i]=nameTst2[i]
    else:
        nameTst2[i]=nameTst2[i][1:] # only keep number, delete first letter
    label_nameTst2[i,1]=nameTst2[i]
    


######Test3
labelN_Tst3=labelTst3.shape[0]

label_nameTst3=np.empty((labelN_Tst3,2),dtype=int) # label + name
for i in range(labelN_Tst3):
    label_nameTst3[i,0]=labelTst3[i] # label
    if (nameTst3[i].isdigit()):
        nameTst3[i]=nameTst3[i]
    else:
        nameTst3[i]=nameTst3[i][1:] # only keep number, delete first letter
    label_nameTst3[i,1]=nameTst3[i]






# split data into train and test groups
train_data,test_data,train_label,test_label = train_test_split(data,label_name,test_size=0.2,random_state=1,stratify=label)


# creat CNN model
def createCNN():
    model = Sequential()
    

    model.add(Convolution2D(128, 3, 3, padding="same", input_shape=(100, 100, 1)))
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
    
    model.add(Dense(2, activation = 'softmax'))
    
    model.summary()
    
    return model



LEARNING_RATE = 0.001
BATCH_SIZE = 16
EPOCHS = 100


model = createCNN()

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(lr=LEARNING_RATE),
              metrics=['accuracy'])





OUT_DIR = "D:/CNNtest2022/OutModel2022"
checkpoint = ModelCheckpoint(os.path.join(OUT_DIR, 'ModelDEM2022.h5'),  # model filename
                              monitor='val_accuracy', # quantity to monitor
                              verbose=0, # verbosity - 0 or 1
                              save_best_only= True, # The latest best model will not be overwritten
                              save_weights_only=False, # save model, not only weights
                              mode='auto') # The decision to overwrite model is made 
                                            # automatically depending on the quantity to monitor

model_details = model.fit(train_data, train_label[:,0],
                          batch_size = BATCH_SIZE,
                          epochs = EPOCHS, 
                          validation_split=0.25,
                          callbacks=[checkpoint],
                          verbose=1)


                                
                                           

scroe, accuracy = model.evaluate(test_data, test_label[:,0], batch_size=BATCH_SIZE)
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
    


confusion_matrix = tf.math.confusion_matrix(test_label[:,0],pred_label, num_classes=None, dtype=tf.int32, name=None, weights=None)
# sess=tf.compat.v1.Session()
sess=tf.Session()

#with tf.compat.v1.Session(graph=g) as sess:
confusion_matrix = sess.run(confusion_matrix)
plot_confusion_matrix(confusion_matrix)




######################### learning curve

def plot_learning_curves(history):
    df=pd.DataFrame(history.history,index=np.arange(0, EPOCHS).astype(dtype=np.str))
    df.plot(use_index=True,figsize=(8, 5))
    plt.grid(True)
   
    #plt.gca().set_ylim(0, 1)
    plt.show()
    
plot_learning_curves(model_details)

print ('Accuracy:', '{:.4f}'.format(accuracy), 'Batch Size:', BATCH_SIZE,'Learning Rate:', LEARNING_RATE, 'Epochs:', EPOCHS)




########################################### Test

################Test1

scroeTst1, accuracyTst1 = model.evaluate(dataTst1,label_nameTst1[:,0], batch_size=BATCH_SIZE)
print('CNN_Test_Test1: loss:', scroeTst1, 'accuracy:', accuracyTst1)

# Predict
pred_labelTst1 = model.predict(dataTst1)
pred_labelTst1 = np.argmax(pred_labelTst1, axis=1)


for t1 in range(pred_labelTst1.shape[0]):
    NameWrongTst1=[]
    labelPredTst1=[]
    if (pred_labelTst1[t1]!=label_nameTst1[t1,0]):
        NameWrongTst1.append(label_nameTst1[t1,1])
        labelPredTst1.append(pred_labelTst1[t1])

np.save("D:/CNNtest2022/ResultName/NameWrongTst1.npy",NameWrongTst1) # names of wrongly classified images
np.save("D:/CNNtest2022/ResultName/LabelPredTst1.npy",labelPredTst1)



# Confusion matrix
confusion_matrix_Tst1 = tf.math.confusion_matrix(label_nameTst1[:,0],pred_labelTst1, num_classes=None, dtype=tf.int32, name=None, weights=None)
# sess=tf.compat.v1.Session()
sess_Tst1=tf.Session()

#with tf.compat.v1.Session(graph=g) as sess:
confusion_matrix_Tst1 = sess_Tst1.run(confusion_matrix_Tst1)
plot_confusion_matrix(confusion_matrix_Tst1)




################Test2

scroeTst2, accuracyTst2 = model.evaluate(dataTst2,label_nameTst2[:,0], batch_size=BATCH_SIZE)
print('CNN_Test_Test2: loss:', scroeTst2, 'accuracy:', accuracyTst2)

# Predict
pred_labelTst2 = model.predict(dataTst2)
pred_labelTst2 = np.argmax(pred_labelTst2, axis=1)


for t2 in range(pred_labelTst2.shape[0]):
    NameWrongTst2=[]
    labelPredTst2=[]
    if (pred_labelTst2[t2]!=label_nameTst2[t2,0]):
        NameWrongTst2.append(label_nameTst2[t2,1])
        labelPredTst2.append(pred_labelTst2[t2])

np.save("D:/CNNtest2022/ResultName/NameWrongTst2.npy",NameWrongTst2) # names of wrongly classified images
np.save("D:/CNNtest2022/ResultName/LabelPredTst2.npy",labelPredTst2)




# Confusion matrix
confusion_matrix_Tst2 = tf.math.confusion_matrix(label_nameTst2[:,0],pred_labelTst2, num_classes=None, dtype=tf.int32, name=None, weights=None)
# sess=tf.compat.v1.Session()
sess_Tst2=tf.Session()

#with tf.compat.v1.Session(graph=g) as sess:
confusion_matrix_Tst2 = sess_Tst2.run(confusion_matrix_Tst2)
plot_confusion_matrix(confusion_matrix_Tst2)



################Test3

scroeTst3, accuracyTst3 = model.evaluate(dataTst3,label_nameTst3[:,0], batch_size=BATCH_SIZE)
print('CNN_Test_Test3: loss:', scroeTst3, 'accuracy:', accuracyTst3)

# Predict
pred_labelTst3 = model.predict(dataTst3)
pred_labelTst3 = np.argmax(pred_labelTst3, axis=1)


for t3 in range(pred_labelTst3.shape[0]):
    NameWrongTst3=[]
    labelPredTst3=[]
    if (pred_labelTst3[t3]!=label_nameTst3[t3,0]):
        NameWrongTst3.append(label_nameTst3[t3,1])
        labelPredTst3.append(pred_labelTst3[t3])

np.save("D:/CNNtest2022/ResultName/NameWrongTst3.npy",NameWrongTst3) # names of wrongly classified images
np.save("D:/CNNtest2022/ResultName/LabelPredTst3.npy",labelPredTst3)




# Confusion matrix
confusion_matrix_Tst3 = tf.math.confusion_matrix(label_nameTst3[:,0],pred_labelTst3, num_classes=None, dtype=tf.int32, name=None, weights=None)
# sess=tf.compat.v1.Session()
sess_Tst3=tf.Session()

#with tf.compat.v1.Session(graph=g) as sess:
confusion_matrix_Tst3 = sess_Tst3.run(confusion_matrix_Tst3)
plot_confusion_matrix(confusion_matrix_Tst3)


