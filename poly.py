# Importing relevant packages
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Declaring the data to be plotted

# Training data
x_train = np.array([14.2, 16.4, 11.9, 15.2, 18.5]).reshape(-1, 1)  # Temperature in degrees Celsius
y_train = np.array([215, 325, 185, 332, 406]).reshape(-1, 1)  # Sales in rands

# Testing data
x_test = np.array([22.1, 19.4, 25.1, 23.4, 18.1, 22.6, 17.2]).reshape(-1, 1)  # Temperature in degrees Celsius
y_test = np.array([522, 412, 614, 544, 421, 445, 408]).reshape(-1, 1)  # Sales in rands

# Training the linear regression model and plotting a prediction
regressor = LinearRegression()
regressor.fit(x_train, y_train)
xx = np.linspace(10, 30, 100)
yy = regressor.predict(xx.reshape(xx.shape[0], 1))
plt.plot(xx, yy)

# Setting the degree of the Polynomial Regression model
quadratic_featuriser = PolynomialFeatures(degree=2)

# This preprocessor transforms an input data matrix into a new data matrix of a given degree
x_train_quadratic = quadratic_featuriser.fit_transform(x_train)
x_test_quadratic = quadratic_featuriser.transform(x_test)

# Training and testing the regressor_quadratic model
regressor_quadratic = LinearRegression()
regressor_quadratic.fit(x_train_quadratic, y_train)
xx_quadratic = quadratic_featuriser.transform(xx.reshape(xx.shape[0], 1))

# Plotting the graph
plt.plot(xx, regressor_quadratic.predict(xx_quadratic), c='r', linestyle='--')
plt.title('Ice Cream Sales vs Temperature')
plt.xlabel('Temperature in Degrees Celsius')
plt.ylabel('Ice Cream Sales in Rands')
plt.scatter(x_train, y_train, c='g')
plt.scatter(x_test, y_test, c='b')
plt.grid(True)
plt.show()