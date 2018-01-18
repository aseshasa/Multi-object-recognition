import os
import PIL
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Input, Flatten, Lambda, SimpleRNN, LSTM, Reshape
from keras.layers.convolutional import Conv2D
from keras.layers.merge import Multiply, Concatenate
from keras import backend as K
from keras.models import Model

def output_of_lambda(input_shape):
    return (input_shape[0], input_shape[1])

def att(ip, classes, loc):
    x1 = Input(shape=ip[0], batch_shape=(1,1,784), name='input1_patch')
    l2 = Input(shape=ip[1], batch_shape=(1,1,2), name='input2_loc')
    x3 =Input(shape=ip[2], batch_shape=(1,1,625), name='input3_coarse')
    # Glimpse layer
    m1 = Dense(256, activation='relu')(x1)
    m2 = Dense(256, activation='relu')(l2)
    g = Multiply()([m1, m2])
    # Recurrent layer
    r1 = LSTM(512, return_sequences=False, activation='tanh', stateful=True)(g)
    rr1 = Reshape((1,512))(r1)
    m = Concatenate(axis=2)([rr1, x3])
    r2 = LSTM(512, return_sequences=False, activation='tanh', stateful=True)(m)
    # Emission layer
    e = Dense(loc, activation=None, name='emit')(r2)
    # Classification layer
    c = Dense(classes, activation='softmax', name='class')(r1)
  
    model = Model(inputs=[x1,l2,x3], outputs=[c,e])
    return model

from keras.datasets import mnist
import numpy as np
from keras.utils import np_utils

def curate_data(x, y, group, loc):
    m = x.shape[0]
    data = np.zeros((m,100,100))
    label = np.zeros((m,2), dtype=np.uint8)
    k = 0 
    for (i,j) in group: 
        new = np.zeros((100,100)) 
        if abs(loc[k,2] - loc[k,0]) < 28: 
            loc[k,2] = (max(loc[k,0],loc[k,2])+28)%(100-29)
        if abs(loc[k,3] - loc[k,1]) < 28:
            loc[k,3] = (max(loc[k,1],loc[k,3])+28)%(100-29)
        new[loc[k,0]:loc[k,0]+28,loc[k,1]:loc[k,1]+28] = x[i,:,:]
        new[loc[k,2]:loc[k,2]+28,loc[k,3]:loc[k,3]+28] = x[j,:,:] 
        data[k,:,:] = new
        label[k,0] = y[i]
        label[k,1] = y[j]
        k += 1
    return (data, label, loc)

import cv2

def predict_on_image(img, model):
    loc_out = np.zeros((11,2), dtype=int)
    lab_out = np.zeros((10,10))
    loc_out[0,:] = np.random.randint(0,100-29,size=(1,2))
    patch = np.reshape(img[loc_out[0,0]:loc_out[0,0]+28,loc_out[0,1]:loc_out[0,1]+28], (1,1,784))
    loc_in = np.reshape(loc_out[0,:], (1,1,2))
    loc_in = loc_in/100-0.5
    for j in range(0,2):
        for k in range(0,5):
            ind = j*5+k
            img_coarse = cv2.resize(img, None, fx=1/4, fy=1/4, interpolation=cv2.INTER_AREA)
            img_coarse = np.reshape(img_coarse, (1,1,625))
            (classes, pred_loc) = model.predict_on_batch({'input1_patch': patch, 'input2_loc': loc_in, 
                                              'input3_coarse': img_coarse})
            loc_in = np.reshape(pred_loc, (1,1,2))
            loc_out[ind+1,:] = (((pred_loc+0.5)*100)%(100-29)).astype(int)
            patch = np.reshape(img[loc_out[ind+1,0]:loc_out[ind+1,0]+28,loc_out[ind+1,1]:loc_out[ind+1,1]+28], (1,1,784))
            lab_out[ind,:] = classes
    model.reset_states()
    return (lab_out, loc_out)

def train_on_image(img, gnd_class, gnd_loc, loc_out, model):
    for j in range(0,2):
        for k in range(0,5):
            ind = j*5+k
            loc_in = loc_out[ind,:]/100-0.5
            patch = np.reshape(img[loc_out[ind,0]:loc_out[ind,0]+28,loc_out[ind,1]:loc_out[ind,1]+28], (1,1,784))
            if j==0:
                true_loc = gnd_loc[0:2]/100-0.5
                true_class = gnd_class[0]
            else:
                true_loc = gnd_loc[2:4]/100-0.5
                true_class = gnd_class[1]
            true_loc = np.reshape(true_loc, (1,2))
            img_coarse = cv2.resize(img, None, fx=1/4, fy=1/4, interpolation=cv2.INTER_AREA)
            img_coarse = np.reshape(img_coarse, (1,1,625))
            loc_in = np.reshape(loc_in, (1,1,2))
            history = model.train_on_batch({'input1_patch': patch, 'input2_loc': loc_in, 
                                 'input3_coarse': img_coarse}, 
                                {'class': np_utils.to_categorical(true_class, 10), 'emit': true_loc})
    model.reset_states()
    return model

def metrics(lab, loc, gnd_lab, gnd_loc):
    label = np.zeros((2,2))
    location = np.zeros((2,2))
    loc_err = np.zeros((1,2))
    lab_err = np.zeros((1,2))
    
    label[0,0] = np.argmax(np.sum(lab[0:5,:], 0))
    label[0,1] = np.mean(lab[0:5,int(label[0,0])])
    label[1,0] = np.argmax(np.sum(lab[5:10,:], 0))
    label[1,1] = np.mean(lab[5:10,int(label[1,0])])
    for i in range(0,5):
        location[0,:] = location[0,:] + lab[i,int(label[0,0])] * loc[i,:]
        location[1,:] = location[1,:] + lab[i+5,int(label[1,0])] * loc[i+5,:]
        
    loc_err[0,0] = np.sqrt(np.sum(np.square(location[0,:]-gnd_loc[0:2])))
    loc_err[0,1] = np.sqrt(np.sum(np.square(location[1,:]-gnd_loc[2:4])))
    lab_err[0,0] = (label[0,0]==gnd_lab[0])
    lab_err[0,1] = (label[1,0]==gnd_lab[1])
    return (label, location, lab_err, loc_err)

def training_epoch(train_data, train_label, train_loc, model):
    lab_err = np.zeros((train_data.shape[0],2))
    loc_err = np.zeros((train_data.shape[0],2))
    for i in range(0,train_data.shape[0]):
        (lab, loc) = predict_on_image(train_data[i,:,:], model)
        (label, location, lab_err[i,:], loc_err[i,:]) = metrics(lab, loc, train_label[i,:], train_loc[i,:])
        model = train_on_image(train_data[i,:,:], train_label[i,:], train_loc[i,:], loc, model)
    return (model, np.mean(np.mean(lab_err, 1)), np.mean(np.mean(loc_err, 1)))
        
def testing_epoch(test_data, test_label, test_loc, model):
    lab_err = np.zeros((test_data.shape[0],2))
    loc_err = np.zeros((test_data.shape[0],2))
    for i in range(0,test_data.shape[0]):
        (lab, loc) = predict_on_image(test_data[i,:,:], model)
        (label, location, lab_err[i,:], loc_err[i,:]) = metrics(lab, loc, test_label[i,:], test_loc[i,:])
    return (np.mean(np.mean(lab_err, 1)), np.mean(np.mean(loc_err, 1)))

(x_train, y_train), (x_test, y_test) = mnist.load_data()
r_tr = np.random.randint(0, 60000-1, size=(60000,2))
r_te = np.random.randint(0, 10000-1, size=(10000,2)) 
l_tr = np.random.randint(0, 100-29, size=(60000,4))
l_te = np.random.randint(0, 100-29, size=(10000,4))

(train_data, train_label, train_loc) = curate_data(x_train, y_train, r_tr, l_tr)
(test_data, test_label, test_loc) = curate_data(x_test, y_test, r_te, l_te)

num_epoch = 10
model = att(((1,784,),(1,2),(1,625)),10,2)
model.compile(optimizer='nadam',
              loss={'class': 'binary_crossentropy', 'emit': 'binary_crossentropy'},
              loss_weights={'class': 1., 'emit': 1.})

train_lab_err = np.zeros((num_epoch,1))
train_loc_err = np.zeros((num_epoch,1))
test_lab_err = np.zeros((num_epoch,1))
test_loc_err = np.zeros((num_epoch,1))
for i in range(0,num_epoch):
    (model, train_lab_err[i], train_loc_err[i]) = training_epoch(train_data, train_label, test_loc, model)
    (test_lab_err[i], test_loc_err[i]) = testing_epoch(test_data, test_label, test_loc, model)

%store test_lab_err > test_lab_err.txt
%store test_loc_err > test_loc_err.txt
%store train_lab_err > train_lab_err.txt
%store train_loc_err > train_loc_err.txt
model.save('my_model.h5')
