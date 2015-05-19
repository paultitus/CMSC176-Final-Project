import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def save(path, ext='png', close=True, verbose=True):
    """Save a figure from pyplot.

    Parameters
    ----------
    path : string
        The path (and filename, without the extension) to save the
        figure to.

    ext : string (default='png')
        The file extension. This must be supported by the active
        matplotlib backend (see matplotlib.backends module).  Most
        backends support 'png', 'pdf', 'ps', 'eps', and 'svg'.

    close : boolean (default=True)
        Whether to close the figure after saving.  If you want to save
        the figure multiple times (e.g., to multiple formats), you
        should NOT close it in between saves or you will have to
        re-plot it.

    verbose : boolean (default=True)
        Whether to print information about when and where the image
        has been saved.

    """
    
    # Extract the directory and filename from the given path
    directory = os.path.split(path)[0]
    filename = "%s.%s" % (os.path.split(path)[1], ext)
    if directory == '':
        directory = '.'
 
    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
 
    # The final path to save to
    savepath = os.path.join(directory, filename)
 
    if verbose:
        print("Saving figure to '%s'..." % savepath),
 
    # Actually save the figure
    plt.savefig(savepath)
    
    # Close it
    if close:
        plt.close()
 
    if verbose:
        print("Done")

# turn of data table rendering
pd.set_option('display.notebook_repr_html', False)

sns.set_palette(['#00A99D', '#F5CA0C', '#B6129F', '#76620C', '#095C57'])
pd.version.version

training_set = pd.read_csv('trainingdata.csv')
training_set.describe()

# Plot the distribution of volumes of the stocks
# sns.distplot(training_set.volume)
# plt.show()



# print features_norm

# 3. Set the initial alpha and number of iterations
alpha = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
iterations = 0
# convergence = 0.0000001

output = open('close results.txt', 'wb')

for num in alpha:
    # convergence_delta = 999

    # 1. Normalize our feature set x
    features = training_set[['open', 'high', 'low', 'close', 'volume_delta']]
    observations = training_set['next_close']
    mu = features.mean()
    sigma = features.std()

    features_norm = (features - mu) / sigma
    features_norm.head()

    # 2. Add a constant feature x0 with value 1
    m = len(features_norm)  # number of data points
    features_norm['x0'] = pd.Series(np.ones(m))
    n = features_norm.columns.size  # number of features
    features_norm.head()

    m = len(observations) * 1.0
    # iterations = 0
    iterations = 0

    thetas = np.zeros(len(features_norm.columns))
    print thetas

    # 5. Calculate the theta's by performing Gradient Descent
    features_norm = np.array(features_norm)
    observations = np.array(observations)

    cost_history = []

    previous_cost = 0

    # while convergence_delta >= convergence:
    while iterations < 10000:
        # Calculate the predicted values
        predicted = np.dot(features_norm, thetas)

        # Calculate the theta's for this iteration:
        thetas -= (num / m) * np.dot((predicted - observations), features_norm)
        
        # Calculate cost
        sum_of_square_errors = np.square(predicted - observations).sum()
        cost = sum_of_square_errors / (2 * m)
        
        # convergence_delta = cost - previous_cost
        # previous_cost = cost
        # Append cost to history
        cost_history.append(cost)
        iterations += 1
        
    print thetas

    # Plot the last 25 entries of the cost history  
    plt.plot(cost_history[:25])
    save("experiments\close\Cost Alpha=" + str(num) + " Iterations=" + str(iterations), ext="png", close=True, verbose=True)
    # plt.show()

    # Calculate the predicted volumes
    training_set['predictions'] = np.dot(features_norm, thetas)
    training_set['difference'] = training_set['predictions'] - training_set['next_close']
    training_set.head()
    print training_set

    # Plot the predicted against the observed values
    p = sns.lmplot("predictions", "next_close", data=training_set, size=7)
    p.set_axis_labels("Predicted Close Next Day", "Actual Close Next Day")
    save("experiments\close\Close Alpha=" + str(num) + " Iterations=" + str(iterations), ext="png", close=True, verbose=True)
    # plt.show()


    # Plot the residuals
    # p = sns.residplot(training_set.predictions, training_set.volume, lowess=True)
    # plt.show()


    # Calculate the coefficient of determination (r^2)
    y = np.array(training_set.next_close)
    p = np.array(training_set.predictions)
    xbar = np.mean(y)

    r_squared = 1 - np.square(y - p).sum() / np.square(y - xbar).sum()
    print r_squared

    mse = ((y - p) ** 2).mean(axis=None) 


    string ="Alpha:" + str(num) + '\n'
    string += "Number of Iterations: " + str(iterations) + '\n'
    # string += "Convergence Point: " + str(convergence) + '\n'
    string += "MSE: " + str(mse) + '\n\n'
    output.write(string)