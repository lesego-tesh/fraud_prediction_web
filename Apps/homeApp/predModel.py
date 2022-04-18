# load and evaluate a saved model
import tensorflow as tf
from tensorflow import keras
import numpy as np
from numpy import loadtxt
from keras.models import load_model
import os
from django.conf import settings
import pandas as pd 
from sklearn.preprocessing  import StandardScaler
 
# load model
path = os.path.join(settings.BASE_DIR, 'media/modfraud.h5')
model = load_model(path)
# summarize model.
model.summary()


class predfunction():

    dataset1 = None
    data_train1 = None
    data_train2 = None  
    data_train3 = None
    data_category = None
    data_gender = None
    data_recency_segment = None
    data_city_pop_segment = None
    data_location = None
    data_f = None



  
  

    def __init__(self, locat) -> None:
       if  locat:
        self.dataset1 = pd.read_csv(locat) 
        

    def predict(self):
        # Convert transform dataset
        # we cannot work on trans_num as there is no unique pattern, so dropping it
        self.dataset1 = self.dataset1.drop("trans_num",1)

        # we can have look on unix time, unix time is generally the number of seconds passed from the UNIX EPOCH 
        # we can use this to know the recency of transactions of same cc_num
        self.dataset1["recency"] = self.dataset1.groupby(by="cc_num")["unix_time"].diff()
        print(self.dataset1.info())
        # checking null values of recency
        
        self.dataset1["recency"].isnull().sum()
        
        # we are getting null values because as 983 because there are 983 unique values of cards, this means whenever the cc_num group changes
        # python makes the first value of every group null, so making them as starting payment, we will initialize null values to -1
       
        self.dataset1.loc[self.dataset1.recency.isnull(),["recency"]] = -1

        # print(self.dataset1.isnull().sum() )
        
       
        # converting trans_date_trans_time to datetime
        self.dataset1["trans_date_trans_time"] = pd.to_datetime(self.dataset1["trans_date_trans_time"])
        
         # Dropping unix time
        self.dataset1 = self.dataset1.drop("unix_time",1)
       
       
        # Unnamed: 0 as it is the index only and we have index present with us
        self.dataset1 = self.dataset1.drop(columns=["Unnamed: 0"])
        
        #dropping dob
       
        self.dataset1 = self.dataset1.drop("dob",1)
        

        # sometimes distance from the customer's home location to the merchant's location can prove out to be main reason for fraud, so taking the 
        # difference of longitude and lattitude of respective columns
        # we have used abs function so that we get proper distance diiference in positive as abs makes negative values positive and used as a mod function

        
        self.dataset1["lat_diff"] = abs(self.dataset1.lat - self.dataset1.merch_lat)
        self.dataset1["long_diff"] = abs(self.dataset1["long"] - self.dataset1["merch_long"])
       
        
        # here we have applied pythogoras theorem and we have multiplied with 110 because each degree of longitude and lattitude 110 kilometers apart
        self.dataset1["displacement"] = np.sqrt(pow((self.dataset1["lat_diff"]*110),2) + pow((self.dataset1["long_diff"]*110),2))
        
       
        # now since we got the displacement so longitudes and lattitudes columns are of no use now, so we can remove them
        self.dataset1 = self.dataset1.drop(columns = ["lat","long","merch_lat","merch_long","lat_diff","long_diff"])

        self.dataset1 = self.dataset1.drop(columns = ["city","zip","street"])
        

        # now we can bin the displacement into near, far and very far records
        # if merchant lies between the range of 0-45 then it is near, while above 45 but below 90 will be far and rest can be very far
        
        self.dataset1.loc[(self.dataset1["displacement"]<45),["location"]] = "Nearby"
        self.dataset1.loc[((self.dataset1["displacement"]>45) & (self.dataset1["displacement"]<90)),["location"]] = "Far Away"
        self.dataset1.loc[(self.dataset1["displacement"]>90),["location"]] = "Long Distance"
       
        # checking location column
      
        print( self.dataset1.location.value_counts(normalize = True))

        # converting Time column to datetime
        self.dataset1["Time"] = pd.to_datetime(self.dataset1["trans_date_trans_time"]).dt.hour
       
        # segregating city_population tab on the basis of less dense, adequately densed, densely populated
        self.dataset1.loc[(self.dataset1["city_pop"]<10000),["city_pop_segment"]] = "Less Dense"
        self.dataset1.loc[((self.dataset1["city_pop"]>10000) & (self.dataset1["city_pop"]<50000)),["city_pop_segment"]] = "Adequately Dense"
        self.dataset1.loc[(self.dataset1["city_pop"]>50000),["city_pop_segment"]] = "Densely populated"

        
        # checking constitution of each segment
        self.dataset1.city_pop_segment.value_counts(normalize = True)

        # dropping column city_pop as it is of no use now
        
        self.dataset1 = self.dataset1.drop("city_pop",1)
       
        # dividing recency column into segments but first converting them from seconds to minutes
        self.dataset1.recency =self.dataset1.recency.apply(lambda x: float((x/60)/60))

        
        # dividing recency to segments based on number of hours passed
        self.dataset1.loc[(self.dataset1["recency"]<1),["recency_segment"]] = "Recent_Transaction"
        self.dataset1.loc[((self.dataset1["recency"]>1) & (self.dataset1["recency"]<6)),["recency_segment"]] = "Within 6 hours"
        self.dataset1.loc[((self.dataset1["recency"]>6) & (self.dataset1["recency"]<12)),["recency_segment"]] = "After 6 hours"
        self.dataset1.loc[((self.dataset1["recency"]>12) & (self.dataset1["recency"]<24)),["recency_segment"]] = "After Half-Day"
        self.dataset1.loc[(self.dataset1["recency"]>24),["recency_segment"]] = "After 24 hours"
        self.dataset1.loc[(self.dataset1["recency"]<0),["recency_segment"]] = "First Transaction"
        self.dataset1.recency_segment.value_counts(normalize = True)
        
        ## create dummy variables
        #creating dummy variables
        self.data_train1 = self.dataset1.drop(columns=["trans_date_trans_time","merchant","job","state"])
        self.data_train2 = pd.get_dummies(self.data_train1,columns=["category","gender","recency_segment","city_pop_segment","location"], drop_first=True)

        # One-Hot Encoding
        # One Hot Encoding is a process in the data processing that is applied to categorical data, 
        # to convert it into a binary vector representation for use in machine learning algorithms
        self.data_train3 = pd.get_dummies(self.data_train1, columns=["category","gender","recency_segment","city_pop_segment","location"], drop_first=True)
        # self.data_train3.info()

        self.data_category = pd.get_dummies(self.data_train1.category, prefix='category')
        # self.data_category.info()

        self.data_gender = pd.get_dummies(self.data_train1.gender, prefix='gender')
        # self.data_gender.info()

        self.data_recency_segment = pd.get_dummies(self.data_train1.recency_segment, prefix='recency_segment')
        # self.data_recency_segment.info()

        self.data_city_pop_segment = pd.get_dummies(self.data_train1.city_pop_segment, prefix='city_pop_segment')
        # self.data_city_pop_segment.info()

        self.data_location = pd.get_dummies(self.data_train1.location, prefix='location')
        # self.data_location.info()

        self.data_f = pd.concat([self.data_train1, self.data_category, self.data_gender, self.data_recency_segment, self.data_city_pop_segment, self.data_location], axis=1)

        ##remove columns that are not needed then later remove the target variable
        self.data_f = self.data_f.drop(columns=['first','last','location','category','recency_segment','city_pop_segment','displacement','gender_F','gender'])
        # self.data_f.info()

        # Get New Shape
        shape = len(self.data_f.columns)

        #model.layers[1].batch_input_shape = shape
        new_model = model.layers.pop(0)
        new_model = tf.keras.Sequential()
        new_model.add(tf.keras.Input(shape=(None,shape)))
        for layer in model.layers[1:]:
         new_model.add(layer)

        # rebuild model architecture by exporting and importing via json
        new_model = keras.models.model_from_json(new_model.to_json())
        new_model.summary()

        result = "fraudulent"
        prediction = new_model.predict(self.data_f)
        if (prediction[0][0]== 0):
            result = "not fraudulent"


        print(new_model.predict(self.data_f))
        return(result)



    def evaluate(self):
        # x_test = pd.read_csv(r"C:\Users\Nkomazana\Desktop\Project\fraudTrain.csv")
        # y_test = pd.

        Evaluation_Results = model.metrics()
        print(Evaluation_Results)
        
        return