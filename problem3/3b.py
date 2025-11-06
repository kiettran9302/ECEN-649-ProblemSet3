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
from scipy.stats import norm
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
h = 0.05
x_min, x_max = -3.0, 9.0
y_min, y_max = -3.0, 9.0

# Regularization values to try
C_list = [0.01, 0.1, 1, 10, 100, 1000]

# Loop through dataset sizes and Cs, fit SVM and save a figure for each
# Create a large independent test set (M = 500 per class) to evaluate classifiers
M = 500
X_test0 = mvn.rvs(mm0, Sig0, M)
X_test1 = mvn.rvs(mm1, Sig1, M)
X_test = np.concatenate((X_test0, X_test1), axis=0)
y_test = np.concatenate((np.zeros(M), np.ones(M)))

results = []

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
        # Evaluate on the large independent test set
        y_pred = clf.predict(X_test)
        errors = np.sum(y_pred != y_test)
        error_rate = errors / float(2 * M)
        results.append({'n': n, 'C': C, 'errors': int(errors), 'error_rate': float(error_rate), 'fname': fname})

# Compute Bayes error for this problem: eps* = Phi(-0.5 * sqrt((m1-m0)^T Sigma^{-1} (m1-m0)))
delta = (mm1 - mm0)
Sigma_inv = np.linalg.inv(Sig0)  # same for both classes here
arg = -0.5 * np.sqrt(float(delta.T.dot(Sigma_inv).dot(delta)))
bayes_error = norm.cdf(arg)

# Save results to CSV and print a sorted table
import csv
out_csv = 'c06_svm_errors.csv'
with open(out_csv, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['n', 'C', 'errors', 'error_rate', 'fname'])
    writer.writeheader()
    for r in results:
        writer.writerow(r)

# Sort results by error_rate ascending
results_sorted = sorted(results, key=lambda x: x['error_rate'])

print('\nBayes error (eps*):', bayes_error)
print('\nTop 5 classifiers by test-set error rate (lowest first):')
print(' rank | n   |    C   | errors | error_rate | filename')
for i, r in enumerate(results_sorted[:5], start=1):
    print(f" {i:>3}  | {r['n']:>3} | {r['C']:>6} | {r['errors']:>6} | {r['error_rate']:.4f} | {r['fname']}")

print(f"\nAll results saved to {out_csv}. Generated {len(results)} classifiers and {len(results)} plots.")
