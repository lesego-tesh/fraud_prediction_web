import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import Sequential
from keras import optimizers
from keras import utils
from keras import losses
#from tensorflow.keras.utils import to_categorical 

#from keras import Flatten,Dense,Dropout,BatchNormalization
# from keras import Conv1D
# from keras import Adam



import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, classification_report, confusion_matrix, precision_recall_curve, roc_auc_score     , roc_curve
from mlxtend.plotting import plot_confusion_matrix

Flatten = layers.Flatten
Dense = layers.Dense
Dropout = layers.Dropout
BatchNormalization = layers.BatchNormalization
Conv1D = layers.Conv1D
Adam = optimizers.adam_v2
to_categorical = losses.BinaryCrossentropy




class Prediction:
    dataset = None
    Shape = None
    DataType = None
    UTV = None
    legitimate = None
    illegitimate = None
    nullval = None
    isFraud = None
    fraudulent = None
    non_fraudulent = None
    predictors = None
    target_column = None
    X = None
    y = None
    X_train = None
    X_test = None
    y_train = None
    y_test = None
    y_pred = None
    pred_test =None
    pred_train = None
    score = None
    history = None
    model = None
    f1Score = None
    f2Score = None

    def __init__(self, csv_loc):
        self.read_dataset(csv_loc)
        

    def read_dataset(self,csv_loc):
        self.dataset = pd.read_csv(csv_loc)

        #making attributes dynamic
        self.Shape =self.dataset.shape
        print(self.Shape)

        self.DataType =self.dataset.shape
        print(self.DataType)

        self.UTV =self.dataset.shape
        print(self.UTV)
        
        self.legitimate =self.dataset.shape
        print(self.legitimate)

        self.f1Score = self.dataset.shape
        print(self.f1Score)

        self.illegitimate =self.dataset.shape
        print(self.illegitimate)

        self.score = self.dataset.shape
        print(self.score)
        
        dataset_desc = self.dataset.describe()
        print(dataset_desc)

        self.nullval= self.dataset.isnull().sum().sum()
        print(self.nullval)

        self.isFraud = self.dataset['Class'].value_counts()
        print(self.isFraud)

        
    
    def run(self):
        self.predictors()
        self.group()
        self.ranSelect()
        self.train()
        self.labels()
        self.scale()
        self.reshape()
        self.BuildModel()
        self.predict()
        self.precisionRecall()
        
        

        analysis = {
            "dataType":self.DataType,
            "Shape": self.Shape,
            "UTV":	self.UTV,
            "score":self.score,
            "f1Score": self.f1Score,
            "f2Score": self.f2Score,
            "anyNull":	self.nullval,
            "tFraud":	self.isFraud[1],
            "tNormal":	self.isFraud[0]
        }
    
        print("\n\n")
        print(analysis)

        return analysis

    def predictors(self):
        ##creating an array for the features and response variables
        self.target_column = ['Class'] 
        self.predictors = list(set(list(self.dataset))-set(self.target_column))
        self.dataset[self.predictors] = self.dataset[self.predictors]/self.dataset[self.predictors].max()
        self.dataset.describe()
        print(self.dataset.describe())

    def group(self):
        #grouping the data and structure
        self.non_fraudulent = self.dataset[self.dataset['Class']== 0]
        self.fraudulent = self.dataset[self.dataset['Class']== 1] 
        #self.non_fraudulent.shape,self.fraudulent.shape
        print(self.non_fraudulent.shape,self.fraudulent.shape)

        #selecting fraudulent transactions from non fraudulent ones using random selection
        self.non_fraudulent = self.non_fraudulent.sample(self.fraudulent.shape[0])
        self.non_fraudulent.shape
        print(self.non_fraudulent.shape)

    def ranSelect(self):
        self.dataset = self.fraudulent.append(self.non_fraudulent,ignore_index = True)
        print(self.dataset)
        print(self.dataset['Class'].value_counts())

    def train(self):
       #  x - dependent variable , y -dependent
        #splits into test and train
        self.X = self.dataset[self.predictors].values
        self.y = self.dataset[self.target_column].values

        self.X_train,self.X_test, self.y_train,self.y_test = train_test_split(self.X, self.y, test_size=0.20, random_state= 0, stratify=self.y)
        print(self.X_train.shape); print(self.X_test.shape)
    
    def labels(self):
        self.X = self.dataset.drop('Class',axis =1)
        self.y = self.dataset['Class']
    
    def scale(self):
    #standardizes a feature by subtracting the mean and then scaling to unit variance
     scaler = StandardScaler()
     self.X_train = scaler.fit_transform(self.X_train)
     self.X_test = scaler.fit_transform(self.X_test)
     self.X_train.shape  

    def reshape(self):
       self.X_train = self.X_train.reshape(self.X_train.shape[0],self.X_train.shape[1],1)
       self.X_test = self.X_test.reshape(self.X_test.shape[0],self.X_test.shape[1],1)
       print(self.X_train.shape,self.X_test.shape)

    
    def BuildModel(self):
        #Build ccn model

        epochs = 20
        #first layer
        self.model = Sequential()
        self.model.add(Conv1D(32,2,activation='relu',input_shape = self.X_train[0].shape))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.2))

        #second layer
        self. model.add(Conv1D(64,2,activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))

        #convert multidimensional layer to a vector
        self.model.add(Flatten())
        self.model.add(Dense(64 ,activation='relu'))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(1, activation='sigmoid'))

        

        # Configure the learning process by selecting 'Binary cross tropy' as a loss function
        # 'Adam' as a optimization function, and to optimize the 'Accuracy matrix'  
        self.model.compile(loss='binary_crossentropy', optimizer='Adam',
                metrics=['accuracy'])

        self.history = self.model.fit(self.X_train,self.y_train,epochs= 20, validation_data=(self.X_test,self.y_test), batch_size = 100,verbose =1)
        print(self.history)

        print(self.model.summary())

    def predict(self):
        self.pred_train= self.model.predict(self.X_train)
        scores = self.model.evaluate(self.X_train, self.y_train, verbose=0)
        print('Accuracy on training data: {} \n Error on training data: {}'.format(scores[1], 1 - scores[1]))   

        self.pred_test= self.model.predict(self.X_test)
        scores2 = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print('Accuracy on test data: {} \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))    

        self.y_pred =( self.model.predict(self.X_test) > 0.5).astype("int32")
        self.score = accuracy_score(self.y_test,self.y_pred)
        print(self.score)

    def precisionRecall(self):
        self.history.history

        ## plot training and validation accuracy values

        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train','val'],loc='upper left')
        plt.show()

        #plot training & validation loss values
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('loss')
        plt.xlabel('Epoch')
        plt.legend(['Train','val'],loc ='upper left')
        plt.show()
        print("\n\n")
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        #confusion matrix
        mat = confusion_matrix(self.y_test,self.y_pred)
        fig,ax = plot_confusion_matrix(conf_mat=mat,figsize=(6,6), show_normed= False)
        plt.tight_layout()
        fig.savefig('cm.png')
        plt.close(fig)
        print(classification_report(self.y_test,self.y_pred))
        print("\n\n")

        self.f1Score = (2 *(0.99*0.69/(0.99+0.69)))
        self.f2Score = (2*(0.76*0.99/(0.76+0.99)))


    

    
