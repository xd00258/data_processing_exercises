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


#---Fitting with linear regression---
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)

#---Fitting with polynomial regression---
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)

#---Visualizing Linear Regression---
plt.scatter(X, Y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Linear Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#---Visalizing Polynomial Regression---
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Polynomial Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#---Predicting new result with Linear Regression---
lin_reg.predict(6.5)

#---Predicting new result with Polynomial Regression---
lin_reg_2.predict(poly_reg.fit_transform(6.5))