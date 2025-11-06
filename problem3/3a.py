"""
Foundations of Pattern Recognition and Machine Learning 2nd edition
Chapter 6 Figure 6.4
Author: Ulisses Braga-Neto
This code is distributed under the GNU LGPL license

Plots SVM classifiers for several regularization (C) values and
several dataset sizes per class. Saves plots to disk.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap

# Fix random state for reproducibility
np.random.seed(1978081)

mm0 = np.array([2, 2])
mm1 = np.array([4, 4])
Sig0 = 4 * np.identity(2)
Sig1 = 4 * np.identity(2)

# List of sample sizes per class to try
N_list = [50, 100, 250, 500]

# Generate datasets for each n. The hint asks that for n=50 the data
# for class 1 be simulated immediately following that for class 0.
# We'll create a list of [X0, X1] pairs for each n.
X_pairs = [[mvn.rvs(mm0, Sig0, n), mvn.rvs(mm1, Sig1, n)] for n in N_list]

cmap_light = ListedColormap(['#ffe0c0', '#b7faff'])

# mesh step size: chosen to keep runtime reasonable while showing
# boundaries over the full range [-3,9] x [-3,9]
h = 0.02
x_min, x_max = -3.0, 9.0
y_min, y_max = -3.0, 9.0

# Regularization values to try
C_list = [0.01, 0.1, 1, 10, 100, 1000]

# Loop through dataset sizes and Cs, fit SVM and save a figure for each
for idx, n in enumerate(N_list):
    X0, X1 = X_pairs[idx]
    x0, y0 = np.split(X0, 2, 1)
    x1, y1 = np.split(X1, 2, 1)
    X = np.concatenate((X0, X1), axis=0)
    y = np.concatenate((np.zeros(n), np.ones(n)))

    for C in C_list:
        clf = SVC(C=C, kernel='rbf', gamma='auto')
        clf.fit(X, y)

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
        plt.rc('xtick', labelsize=12)
        plt.rc('ytick', labelsize=12)

        # plot training points with smaller marker size for clarity
        plt.plot(x0, y0, '.r', markersize=8)  # class 0
        plt.plot(x1, y1, '.b', markersize=8)  # class 1

        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')
        ax.contour(xx, yy, Z, colors='black', linewidths=0.5)

        title = f"SVM RBF (C={C}, n={n} per class)"
        plt.title(title)

        fname = f'c06_svm_n{n}_C{C}.png'
        fig.savefig(fname, bbox_inches="tight", facecolor="white")
        plt.close(fig)

print('Done: generated plots for C in', C_list, 'and n in', N_list)
