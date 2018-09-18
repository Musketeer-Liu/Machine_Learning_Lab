import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


from supervised_learning import logistic_regression
from utilities.data_manipulation import train_test_split, make_diagonal, normalize, accuracy_score
from utilities import plotting




def main():
    data = load_iris()
    X = normalize(data.data[data.target != 0])
    y = data.target[data.target != 0]
    y[y == 1], y[y == 2] = 0, 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, seed=1)

    model = logicstic_regression.LogistcRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred = np.reshape(y_pred, y_test.shape)

    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy: ', accuracy)

    # Reduce to 2D via PCA and plot results
    plotting.plot_in_2d(X_test, y_pred, title='Logistic Regression', accuracy=accuracy)




if __name__ == '__main__':
    main()


