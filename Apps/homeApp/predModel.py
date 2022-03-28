import tensorflow as tf
from tensorflow import keras
import tensorflow as tf
from tensorflow import keras
from . import layers
from . import Sequential
from . import Flatten,Dense,Dropout,BatchNormalization
from . import Conv1D
from . import Adam
from . import to_categorical 
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
    dataset = None

    def __init__(self, csv_loc) -> None:
        self.read_dataset(self,csv_loc)

    def read_dataset(self,csv_loc):
        self.dataset = pd.read_csv(csv_loc)
        print(self.dataset.shape)
        self.dataset.describe()

        dataset_desc = self.dataset.describe()
        print(dataset_desc)
