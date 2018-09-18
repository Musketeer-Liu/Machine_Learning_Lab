import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression


from supervised_learning import linear_regression
from utilities.data_manipulation import train_test_split
from utilities.data_operation import mean_squared_error
from utilities import plotting




def main():
    X, y = make_regression(n_samples=100, n_features=1, noise=20)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    n_samples, n_features = np.shape(X)

    # Setup & Train a Linear_Regression Model
    model = linear_regression.LinearRegression(n_iterations=3000, regularization=l2_regularization(alpha=0.5))
    model.fit(X_train, y_train)

    # Plot Loss Diagram
    n = len(model.training_errors)
    training, = plt.plot(range(n), model.training_errors, label="Training Error")
    plt.legend(handlers=[training])
    plt.title('Error Plot')
    plt.ylabel("Mean Squared Error")
    plt.xlabel("Iterations")
    plt.show()

    # Predict Model
    y_pred = model.predict(X_test)
    
    # Evaluate Model
    y_pred_reshape = np.reshape(y_pred, y_test.shape)
    mse = mean_squared_error(y_test, y_pred_reshape)
    print("Mean Squared Error")

    # Predict Model again
    y_pred = model.predict(X_test)

    # Plot Color Map
    cmap = plt.get_cmap('viridis')

    # Plot fitting results
    m1 = plt.scatter(366*X_train, y_train, colol=cmap(0.9), s=10)
    m2 = plt.scatter(366*X_test, y_test, color=cmap(*0.5), s=10)
    plt.plot(366*X, y_pred, color='black')
    
    plt.suptitle('Linear Regression')
    plt.title("MSE: {:.2f}".format(mse), fontsize=10)
    plt.xlabel('Day')
    plt.ylabel('Temperature in Celcius')
    plt.legend((m1, m2), ('Training Data', 'Test Data'), loc='lower right')
    
    plt.show()




if __name__ == '__main__':
    main()
