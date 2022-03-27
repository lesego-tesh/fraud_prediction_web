import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten,Dense,Dropout,BatchNormalization
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score     , fbeta_score, classification_report, confusion_matrix, precision_recall_curve, roc_auc_score     , roc_curve
from mlxtend.plotting import plot_confusion_matrix

class Prediction:
    df_fraud = None

    def __init__(self, csv_loc) -> None:
        self.read(self,csv_loc)

    def read(self,csv_loc):
        self.df_fraud = pd.read_csv(csv_loc)
        print(self.df_fraud.shape)
        self.df_fraud.describe()
