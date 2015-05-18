import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# turn of data table rendering
pd.set_option('display.notebook_repr_html', False)

sns.set_palette(['#00A99D', '#F5CA0C', '#B6129F', '#76620C', '#095C57'])
pd.version.version

training_set = pd.read_csv('trainingdata.csv')
training_set.describe()

# Plot the distribution of volumes of the stocks
# sns.distplot(training_set.volume)
# plt.show()

# 1. Normalize our feature set x
features = training_set[['open', 'high', 'low', 'close']]
observations = training_set['volume']
mu = features.mean()
sigma = features.std()

features_norm = (features - mu) / sigma
features_norm.head()

# 2. Add a constant feature x0 with value 1
m = len(features_norm)  # number of data points
features_norm['x0'] = pd.Series(np.ones(m))
n = features_norm.columns.size  # number of features
features_norm.head()

# print features_norm

# 3. Set the initial alpha and number of iterations
alpha = 0.25
iterations = 150
m = len(observations) * 1.0

# 4. Initialize the theta values to zero
thetas = np.zeros(len(features_norm.columns))
print thetas

# 5. Calculate the theta's by performing Gradient Descent
features_norm = np.array(features_norm)
observations = np.array(observations)

cost_history = []

for i in range(iterations):
    # Calculate the predicted values
    predicted = np.dot(features_norm, thetas)

    # Calculate the theta's for this iteration:
    thetas -= (alpha / m) * np.dot((predicted - observations), features_norm)
    
    # Calculate cost
    sum_of_square_errors = np.square(predicted - observations).sum()
    cost = sum_of_square_errors / (2 * m)

    # Append cost to history
    cost_history.append(cost)
    
print thetas

# Plot the last 25 entries of the cost history  
plt.plot(cost_history[:25])
plt.show()

# Calculate the predicted brainweights and differences from the observed values
training_set['predictions'] = np.dot(features_norm, thetas)
training_set['difference'] = training_set['predictions'] - training_set['volume']
training_set.head()
print training_set

# Plot the predicted against the observed values
p = sns.lmplot("predictions", "volume", data=training_set, size=7)
p.set_axis_labels("Predicted Volume", "Actual Volume")
plt.show()


# Plot the residuals
p = sns.residplot(training_set.predictions, training_set.volume, lowess=True)
plt.show()


# Calculate the coefficient of determination (r^2)
y = np.array(training_set.volume)
p = np.array(training_set.predictions)
xbar = np.mean(y)

r_squared = 1 - np.square(y - p).sum() / np.square(y - xbar).sum()
print r_squared