import progressbar
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import numpy as np


from tools.data_operation import calculate_covariance_matrix
from tools.data_operation import calculate_correlation_matrix
from tools.data_manipulation import standardize




bar_widgets = [
    'Training: ', 
    progressbar.Percentage(), ' ', 
    progressbar.Bar(marker="-", left="[", right="]"), ' ',
    progressbar.ETA()
]


class Plot():
    def __init__(self):
        self.cmap = plt.get_cmap('viridis')

    # Calculate eigenvalues and eigenvectors
    def _transform(self, X, dim):
        # Calculate eigenvalue and eigenvectors
        covariance = calculate_covariance_matrix(X)
        eigenvalues, eigenvectors = np.linalg.eig(covariance)

        # Sort eigenvalues and eigenvector by largest eigenvalues
        idx = eigenvalues.argsort[::-1]
        eigenvalues = eigenvalues[idx][:dim]
        eigenvectors = np.atleast_1d(eigenvectors[:, idx])[:, :dim]

        # Project the data onto principal components
        X_transformed = X.dot(eigenvectors)
        return X_transformed

    # Plot the dataset X and the corresponding labels in basic 1D
    def plot_regression(self, lines, title,
                        axis_labels=None, mse=None, scatter=None,
                        legend={"type": "lines", "loc": "lower right"}):
        if scatter:
            scatter_plots, scatter_labels = [], []
            for s in scatter:
                scatter_plots += [plt.scatter(s["x"], s["y"], color=s["color"], s=s["size"])]
                scatter_labels += [s["label"]]
            scatter_plots, scatter_labels = tuple(scatter_plots), tuple(scatter_labels)

        for l in lines:
            l = plt.plot(l["x"], l["y"],
                          color=s["color"],
                          linewidth=l["width"],
                          label=l["label"])
        
        if mse:
            plt.suptitle(title)
            plt.title("MSE: {:.2f}".format(mse), fontsize=10)
        else:
            plt.title(title)
        
        if axis_labels:
            plt.xlabel(axis_labels["x"])
            plt.ylabel(axis_labels["y"])
        
        if legend["type"] == "lines":
            plt.legend(loc="lower_left")
        elif legend["type"] == "scatter" and scatter:
            plt.legend(scatter_plots, scatter_labels, loc=legend["loc"])

        plt.show()

    # Plot the dataset X and corresponding labels y in 2D using PCA
    def plot_in_2d(self, X, y=None, title=None, accuracy=None, legend_labels=None):
        X_transformed = self._transform(X, dim=2)
        x1 = X_transformed[:, 0]
        x2 = X_transformed[:, 1]
        class_distr = []
        y = np.array(y).astype(int)
        colors = [self.cmap(i) for i in np.linspace(0, 1, len(np.unique(y)))]

        # Plot differert class distributions
        for i, l in enumerate(np.unique(y)):
            _x1 = x1[y==l]
            _x2 = x2[y==l]
            _y = y[y==l]
            class_distr.append(plt.scatter(_x1, _x2, color=colors[i]))
        
        # Plot legend
        if legend_labels:
            plt.legend(class_distr, legend_labels, loc=1)
        
        # Plot title
        if title:
            if accuracy:
                perc = 100 * accuracy
                plt.suptitle(title)
                plt.title("Accuracy: {:.1f}%".format(perc), fontsize=10)
            else:
                plt.title(title)
        
        # Axis labels
        plt.xlabel('Principal Componenet 1')
        plt.ylabel('Principal Componenet 2')

        plt.show()

    # Plot the dataset X and corresponding labels y in 2D using PCA
    def plot_in_3d(self, X, y=None):
        X_transformed = self._transform(X, dim=3)
        x1 = X_transformed[:, 0]
        x2 = X_transformed[:, 1]
        x3 = X_transformed[:, 2]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x1, x2, x3, c=y)

        plt.show()

            





            