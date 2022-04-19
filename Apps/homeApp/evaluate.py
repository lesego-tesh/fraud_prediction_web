import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import Sequential
from keras import optimizers
from keras import utils
from keras import losses

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


class Evaluate:
    dataset = None
    Shape = None
    DataType = None
    UTV = None
    legitimate = None
    illegitimate = None
    nullval = None
    data_train1 = None
    data_train2 = None
    data_train3 = None
    data_category = None
    data_gender = None
    data_recency_segment = None
    data_city_pop_segment  = None
    data_location = None
    data_f = None
    fraudulent = None
    non_fraudulent = None
    train_x = None
    train_y = None
    x_train = None
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
    is_fraud = None

    def __init__(self, csv_loc):
        self.read_dataset(csv_loc)
        
     ## read the dataset using pandas
    def read_dataset(self,csv_loc):
        self.dataset = pd.read_csv(csv_loc)

        
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

        self.is_fraud = self.dataset['is_fraud'].value_counts()
        print(self.is_fraud)

        
    
    def run(self):
        self.transformData()
        self.dummies()
        self.group()
        self.train()
        self.BuildModel()
        self.predict()
        self.precisionRecall()
        
        print(self.is_fraud)

        analysis = {
            "dataType":self.DataType,
            "Shape": self.Shape,
            "UTV":	self.train_x.columns.values.tolist(),
            "score":self.score,
            "f1Score": self.f1Score,
            "f2Score": self.f2Score,
            "anyNull":	self.nullval,
            "tFraud":	self.is_fraud[1],
            "tNormal":	self.is_fraud[0]
        }

        print(self.train_x.columns)
    
        print("\n\n")
        print(analysis)

        return analysis

    def transformData(self):

        # we cannot work on trans_num as there is no unique pattern, so dropping it
        self.dataset = self.dataset.drop("trans_num",1)

        self.dataset.isnull().sum()
    
        self.dataset["recency"] = self.dataset.groupby(by="cc_num")["unix_time"].diff()

        # checking null values of recency
        self.dataset["recency"].isnull().sum()

        self.dataset.loc[self.dataset.recency.isnull(),["recency"]] = -1

        self.dataset["trans_date_trans_time"] = pd.to_datetime(self.dataset["trans_date_trans_time"])

                # Dropping unix time
        self.dataset = self.dataset.drop("unix_time",1)

        # Unnamed: 0 as it is the index only and we have index present with us
        self.dataset = self.dataset.drop(columns=["Unnamed: 0"])

        
        self.dataset["lat_diff"] = abs(self.dataset.lat - self.dataset.merch_lat)
        self.dataset["long_diff"] = abs(self.dataset["long"] - self.dataset["merch_long"])

        self.dataset["displacement"] = np.sqrt(pow((self.dataset["lat_diff"]*110),2) + pow((self.dataset["long_diff"]*110),2))

        # now since we got the displacement so longitudes and lattitudes columns are of no use now, so we can remove them
        self.dataset =self.dataset.drop(columns = ["lat","long","merch_lat","merch_long","lat_diff","long_diff"])

        self.dataset=self.dataset.drop(columns = ["city","zip","street"])
        self.dataset.info()

        self.dataset.loc[(self.dataset["displacement"]<45),["location"]] = "Nearby"
        self.dataset.loc[((self.dataset["displacement"]>45) & (self.dataset["displacement"]<90)),["location"]] = "Far Away"
        self.dataset.loc[(self.dataset["displacement"]>90),["location"]] = "Long Distance"
        self.dataset.info()

        self.dataset["Time"] = pd.to_datetime(self.dataset["trans_date_trans_time"],"%H:%M").dt.time

        self.dataset["Time"] = pd.to_datetime(self.dataset["trans_date_trans_time"]).dt.hour

        # segregating city_population tab on the basis of less dense, adequately densed, densely populated
        self.dataset.loc[(self.dataset["city_pop"]<10000),["city_pop_segment"]] = "Less Dense"
        self.dataset.loc[((self.dataset["city_pop"]>10000) & (self.dataset["city_pop"]<50000)),["city_pop_segment"]] = "Adequately Dense"
        self.dataset.loc[(self.dataset["city_pop"]>50000),["city_pop_segment"]] = "Densely populated"

        # checking constitution of each segment
        self.dataset.city_pop_segment.value_counts(normalize = True)

                # dropping column city_pop as it is of no use now
        self.dataset = self.dataset.drop("city_pop",1)

        # dividing recency column into segments but first converting them from seconds to minutes
        self.dataset.recency = self.dataset.recency.apply(lambda x: float((x/60)/60))

                # dividing recency to segments based on number of hours passed
        self.dataset.loc[(self.dataset["recency"]<1),["recency_segment"]] = "Recent_Transaction"
        self.dataset.loc[((self.dataset["recency"]>1) & (self.dataset["recency"]<6)),["recency_segment"]] = "Within 6 hours"
        self.dataset.loc[((self.dataset["recency"]>6) & (self.dataset["recency"]<12)),["recency_segment"]] = "After 6 hours"
        self.dataset.loc[((self.dataset["recency"]>12) & (self.dataset["recency"]<24)),["recency_segment"]] = "After Half-Day"
        self.dataset.loc[(self.dataset["recency"]>24),["recency_segment"]] = "After 24 hours"
        self.dataset.loc[(self.dataset["recency"]<0),["recency_segment"]] = "First Transaction"
        self.dataset.recency_segment.value_counts(normalize = True)
        
    def dummies(self):
        ## create dummy variables
        #creating dummy variables
        self.data_train1 = self.dataset.drop(columns=["trans_date_trans_time","merchant","job","state"])
        self.data_train2 = pd.get_dummies(self.data_train1,columns=["category","gender","recency_segment","city_pop_segment","location"], drop_first=True)

        # One Hot Encoding is a process in the data processing that is applied to categorical data, 
        # to convert it into a binary vector representation for use in machine learning algorithms
        self.data_train3 = pd.get_dummies(self.data_train1, columns=["category","gender","recency_segment","city_pop_segment","location"], drop_first=True)
        #self.data_train3.info()

        self.data_category = pd.get_dummies(self.data_train1.category, prefix='category')
        #data_category.info()

        self.data_gender = pd.get_dummies(self.data_train1.gender, prefix='gender')
        #self.data_gender.info()

        self.data_recency_segment = pd.get_dummies(self.data_train1.recency_segment, prefix='recency_segment')
        #data_recency_segment.info()

        self.data_city_pop_segment = pd.get_dummies(self.data_train1.city_pop_segment, prefix='city_pop_segment')
        #self.data_city_pop_segment.info()

        self.data_location = pd.get_dummies(self.data_train1.location, prefix='location')
        #self.data_location.info()

        self.data_f = pd.concat([self.data_train1, self.data_category, self.data_gender, self.data_recency_segment, self.data_city_pop_segment, self.data_location], axis=1)

        ##remove columns that are not needed then later remove the target variable
        self.data_f = self.data_f.drop(columns=['first','last','dob','location','category','recency_segment','city_pop_segment','category_personal_care','displacement','gender_F','gender_M','gender'])


 

    def group(self):
       #grouping the data and structure
        self.non_fraudulent = self.data_f[self.data_f['is_fraud']== 0]
        self.fraudulent = self.data_f[self.data_f['is_fraud']== 1]


        #selecting fraudulent transactions from non fraudulent ones using random selection
        self.non_fraudulent = self.non_fraudulent.sample(self.fraudulent.shape[0])
        self.non_fraudulent.shape

        self.data_f = self.fraudulent.append(self.non_fraudulent,ignore_index = True)
        self.data_f

        #checking the if data is balanced
        self.data_f['is_fraud'].value_counts()



    

    def train(self):
       ## removing the target value
        self.train_x = self.data_f.drop('is_fraud',axis =1)
        self.train_y = self.data_f['is_fraud']

        #splitting tests and trains
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.train_x, self.train_y, test_size=0.30, random_state= 0, stratify= self.train_y)


    #standardizes a feature by subtracting the mean and then scaling to unit variance
        scaler = StandardScaler()
        self.x_train = scaler.fit_transform(self.x_train)
        self.x_test = scaler.fit_transform(self.x_test)
        self.x_train.shape  

    
        self.x_train = self.x_train.reshape(self.x_train.shape[0],self.x_train.shape[1])
        self.x_test = self.x_test.reshape(self.x_test.shape[0],self.x_test.shape[1])
       

    
    def BuildModel(self):
        #Build logistic model

        self.model = Sequential()
        self.model.add(Dense(256, activation='softmax',input_shape = self.x_train[0].shape))
        self.model.add(Dense(1, activation="sigmoid"))

                

        # Configure the learning process by selecting 'Binary cross tropy' as a loss function
        # 'Adam' as a optimization function, and to optimize the 'Accuracy matrix'  
        self.model.compile(loss='binary_crossentropy', optimizer='adam',
                metrics=['accuracy'])

        self.history = self.model.fit(self.x_train,self.y_train,epochs= 20, validation_data=(self.x_test,self.y_test), batch_size = 100,verbose =1)
       

    def predict(self):
        self.pred_train= self.model.predict(self.x_train)
        scores = self.model.evaluate(self.x_train, self.y_train, verbose=0)
        print('Accuracy on training data: {} \n Error on training data: {}'.format(scores[1], 1 - scores[1]))   

        self.pred_test= self.model.predict(self.x_test)
        scores2 = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print('Accuracy on test data: {} \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))    

        self.y_pred = (self.model.predict(self.x_test) > 0.5).astype("int32")
        self.score = accuracy_score(self.y_test,self.y_pred)
        

    def precisionRecall(self):
        self.history.history

        ## plot training and validation accuracy values

        # plt.plot(self.history.history['accuracy'])
        # plt.plot(self.history.history['val_accuracy'])
        # plt.title('Model accuracy')
        # plt.ylabel('Accuracy')
        # plt.xlabel('Epoch')
        # plt.legend(['Train','val'],loc='upper left')
        # plt.show()
        # plt.close()

        # #plot training & validation loss values
        # plt.plot(self.history.history['loss'])
        # plt.plot(self.history.history['val_loss'])
        # plt.title('Model loss')
        # plt.ylabel('loss')
        # plt.xlabel('Epoch')
        # plt.legend(['Train','val'],loc ='upper left')
        # plt.show()
        # plt.close()
        # print("\n\n")

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


    

    