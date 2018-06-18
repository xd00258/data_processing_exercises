"""
Polynomial "Linear" Regression
y = b0 + b1x1 + b2x1^2 + ... + bnx1^n
"""

#------Preprocess------

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values 

#no encoding needed

#no need to split data because need info from everything

#no feature scaling needed


