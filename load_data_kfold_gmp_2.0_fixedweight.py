"""
    This file is part of SequenceCNN.

    SequenceCNN is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    SequenceCNN is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Nome-Programma.  If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as np
from keras.preprocessing import sequence
from keras.layers import Input, Dense, Conv1D, MaxPooling1D,GlobalMaxPooling1D, UpSampling2D, Flatten, Reshape, Dropout, GlobalAveragePooling1D
from keras.models import Model, Sequential
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from collections import Counter
import keras.backend as K
def mcc(y_true, y_pred):
    #tf_session = K.get_session()
    #print(np.array(y_true.eval(session=tf_session)))
    p,u,n,o =0.0,0.0,0.0,0.0

    for pred,tr in zip(y_pred,y_true):
        if tr==1:
	    if pred==1:
		p+=1
	    else:
		u+=1
        else:
	    #tr==0
	    if pred==0:
		n+=1
	    else:
		o+=1

    mcc = ((p*n)-(u*o))/np.sqrt((p+u)*(p+o)*(n+u)*(n+o))
    sensitivity=p/(p+u)
    specificity=n/(n+o)
    return mcc, sensitivity, specificity

def get_class_weights(y):
    counter = Counter(y)
    majority = max(counter.values())
    #weights=  {cls: float(majority/count) for cls, count in counter.items()}
    weights={0:1.0, 1:50.0}
    print "class weights", weights
    return weights	
#variables (will be validated)
import sys
if len(sys.argv)<4:
        sys.exit("python load_data_kfold.py filters_first_layer filters n_layers hidden_dims dataset [TMBB2, Boctopus]")
filtersfirst=int(sys.argv[1])
filters = int(sys.argv[2])
nlayers = int(sys.argv[3])

hidden_dims= int(sys.argv[4])
dataset=sys.argv[5]
#nlayers = 2
#filters = 250 #500 meglio
kernel_size = 5
#hidden_dims = 100
epochs = 40
batch_size = 32
cv_mcc, cv_sens,cv_spec= [],[],[]
for fold in xrange(10):

    if dataset=="TMBB2":	
	prefix="TMBB2/"
    	train_file= prefix+"sets/LR_"+str(fold)#+".name"
    	val_file =prefix+"sets/VL_"+str(fold)#+".name"

    	test_file =prefix+"sets/TS_"+str(fold)#+".name"

    	train_ex_name, train_target=np.split(np.loadtxt(train_file,np.str),2, 1)
    	val_ex_name, val_target= np.split(np.loadtxt(val_file,np.str),2, 1)

    	test_ex_name, test_target= np.split(np.loadtxt(test_file,np.str),2, 1)
    elif dataset=="Boctopus":
	prefix="Boctopus/"
        train_file= prefix+"sets/LR_"+str(fold)#+".name"
        val_file =prefix+"sets/VL_"+str(fold)#+".name"

        test_file =prefix+"sets/TS_"+str(fold)#+".name"

        train_ex_name, train_target=np.split(np.loadtxt(train_file,np.str),2, 1)
        val_ex_name, val_target= np.split(np.loadtxt(val_file,np.str),2, 1)

        test_ex_name, test_target= np.split(np.loadtxt(test_file,np.str),2, 1)

    y_train=[1 if i=="1" else 0for i in train_target ]
    y_val=[1 if i=="1" else 0 for i in val_target]
    y_test=[1 if i=="1" else 0 for i in test_target]


    #test
    y_train=y_train[:]
    #print test_ex_name.shape, test_target.shape

    #consider maximum length in the training set
    maxlen=0
    x_train=[]
    for data_fname in train_ex_name[:]:
        data_fname=data_fname[0]
        temp=np.loadtxt(prefix+"data/"+str(data_fname)+".dat")
        x_train.append(temp)
        length=temp.shape[0]
        if length>maxlen:
            maxlen=length
    
    x_val=[]
    for data_fname in val_ex_name:
        data_fname=data_fname[0]
        temp=np.loadtxt(prefix+"data/"+str(data_fname)+".dat")
        x_val.append(temp)
    
    x_test=[]
    for data_fname in test_ex_name:
        data_fname=data_fname[0]
        temp=np.loadtxt(prefix+"data/"+str(data_fname)+".dat")
        x_test.append(temp)

    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_val = sequence.pad_sequences(x_val, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_val shape:', x_val.shape)
    print('x_test shape:', x_test.shape)

    #end for of 

    n_features=x_train.shape[2]
    #make model
    inputs = Input(shape=(maxlen,n_features,)) 
    dr0=Dropout(0.25)(inputs)
    conv0=Conv1D(filtersfirst,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1)(dr0)
    #mp0= MaxPooling1D(pool_size=3)(conv0) 
    
    for l in range(2,nlayers+1): 
        conv0=Conv1D(filters/(2**(nlayers-l)),                                  kernel_size,
                                 padding='valid',
                                 activation='relu',
                                 strides=1)(conv0)
        #mp0= MaxPooling1D(pool_size=3)(conv0) 
                                 
    #mp1= MaxPooling1D(pool_size=3)(conv1) 

    #mp = GlobalAveragePooling1D()(conv1)             
    mp= GlobalMaxPooling1D()(conv0)
    #mp= MaxPooling1D(pool_size=2)(conv0)  
    #mp = GlobalAveragePooling1D()(conv0)             
    #mp= GlobalMaxPooling1D()(conv1)
    #fl0=Flatten()(mp0)
    
    dr1=Dropout(0.25)(mp)

    d1=Dense(hidden_dims, activation='relu')(dr1) #hidden_dims
    #dr2=Dropout(0.25)(d1)
    #d2=Dense(hidden_dims, activation='relu')(dr2) #hidden_dims
    #dr3=Dropout(0.25)(d1)
    #d3=Dense(hidden_dims, activation='relu')(dr3) #hidden_dims

    outputs=Dense(1,activation='sigmoid')(d1) 

    #create model
    model = Model(input=inputs, output=outputs)
    model.summary()
    model.compile(loss='binary_crossentropy',
              optimizer='Nadam',
              metrics=['accuracy'])

    filepath = dataset+"."+str(fold)+"."+str(filtersfirst)+"."+str(filters)+"."+str(nlayers)+"."+str(hidden_dims)+".TEST.BDtest.weightsautoencoder.best.hdf5"

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=0, mode='auto', epsilon=0.0001,
                                      cooldown=0, min_lr=0)

    model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_val, y_val), callbacks=[early_stopping,lr_reducer,checkpoint],class_weight='auto')#get_class_weights(y_train))
    model.load_weights(filepath)
    y_pred=model.predict(x_test)
    #np.round(y_pred)
    y_pred=(y_pred > 0.5).astype('int32')
    #print y_pred
    #y_pred=np_utils.probas_to_classes(y_pred)
    mcc_fold, sens_fold,spec_fold= mcc(y_test, y_pred)
    cv_mcc.append(mcc_fold)
    cv_sens.append(sens_fold)
    cv_spec.append(spec_fold)
    print "FOLD",str(fold)," MCC, sensitivity, specificity:", mcc(y_test, y_pred)
print "AVERAGE MCC, sensitivity, specificity:", np.mean(cv_mcc), np.std(cv_mcc),np.mean(cv_sens), np.std(cv_sens),np.mean(cv_spec), np.std(cv_spec)
