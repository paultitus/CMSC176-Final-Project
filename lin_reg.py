import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import linear_model

# Loading the training dataset
training_data = np.genfromtxt("trainingdata.csv", delimiter=",", dtype = float)
training_data_X = training_data[:, [0,1,2,3]]
training_data_Y = training_data[:, 4]

# Loading the test dataset
test_data = np.genfromtxt("validationdata.csv", delimiter=",", dtype = float)
test_data_X = test_data[:, [0,1,2,3]]
test_data_Y = test_data[:, 4]

# Creating the linear regression object
regr = linear_model.LinearRegression(fit_intercept=False)

# Training the model using the training set
regr.fit(training_data_X, training_data_Y)

# The coefficients
print('Coefficients: \n', regr.coef_)

# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((regr.predict(test_data_X) - test_data_Y) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(test_data_X, test_data_Y))
