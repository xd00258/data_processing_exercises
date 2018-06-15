#Linear Regression

#--PREPROCESS DATA--

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values #remove salary column for independent (matrix)
Y = dataset.iloc[:, 1].values #get salary column for dependent (vector)

#split dataset in training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

#no feature scaling needed

