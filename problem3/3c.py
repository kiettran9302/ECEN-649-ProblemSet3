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

# We'll perform R repetitions per (n,C) to estimate expected error rates
R = 50

# aggregate results per (n,C)
agg_results = []

for idx, n in enumerate(N_list):
    # keep one visualization dataset per n (as in part a)
    X0_vis, X1_vis = X_pairs[idx]
    x0_vis, y0_vis = np.split(X0_vis, 2, 1)
    x1_vis, y1_vis = np.split(X1_vis, 2, 1)

    for C in C_list:
        err_rates = []
        for r in range(R):
            # generate fresh training set for this repetition
            X0 = mvn.rvs(mm0, Sig0, n)
            X1 = mvn.rvs(mm1, Sig1, n)
            X = np.concatenate((X0, X1), axis=0)
            y = np.concatenate((np.zeros(n), np.ones(n)))

            clf = SVC(C=C, kernel='rbf', gamma='auto')
            clf.fit(X, y)

            # save decision-boundary visualization using the stored visualization dataset
            if r == 0:
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
                Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)

                fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
                plt.rc('xtick', labelsize=12)
                plt.rc('ytick', labelsize=12)
                plt.plot(x0_vis, y0_vis, '.r', markersize=8)  # class 0 (visualization set)
                plt.plot(x1_vis, y1_vis, '.b', markersize=8)  # class 1 (visualization set)
                plt.xlim([x_min, x_max])
                plt.ylim([y_min, y_max])
                plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')
                ax.contour(xx, yy, Z, colors='black', linewidths=0.5)
                title = f"SVM RBF (C={C}, n={n} per class)"
                plt.title(title)
                fname = f'c06_svm_n{n}_C{C}.png'
                fig.savefig(fname, bbox_inches="tight", facecolor="white")
                plt.close(fig)

            # evaluate on the common test set
            y_pred = clf.predict(X_test)
            errors = np.sum(y_pred != y_test)
            error_rate = errors / float(2 * M)
            err_rates.append(error_rate)

        mean_err = float(np.mean(err_rates))
        std_err = float(np.std(err_rates, ddof=0))
        agg_results.append({'n': n, 'C': C, 'mean_error': mean_err, 'std_error': std_err, 'R': R, 'fname': fname})

print(f"Done: repeated experiments completed (R={R})")

# b) Compute Bayes error for this problem: eps* = Phi(-0.5 * sqrt((m1-m0)^T Sigma^{-1} (m1-m0)))
delta = (mm1 - mm0)
Sigma_inv = np.linalg.inv(Sig0)
arg = -0.5 * np.sqrt(float(delta.T.dot(Sigma_inv).dot(delta)))
bayes_error = norm.cdf(arg)

# c) Save aggregated results to CSV
import csv
out_csv = 'c06_svm_agg_errors.csv'
with open(out_csv, 'w', newline='') as csvfile:
    fieldnames = ['n', 'C', 'mean_error', 'std_error', 'R', 'fname']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for r in agg_results:
        writer.writerow(r)

# Prepare data for plotting
import math
N_vals = N_list
# error vs n for each C
plt.figure(figsize=(8, 6), dpi=150)
for C in C_list:
    means = [next(item['mean_error'] for item in agg_results if item['n'] == n and item['C'] == C) for n in N_vals]
    stds = [next(item['std_error'] for item in agg_results if item['n'] == n and item['C'] == C) for n in N_vals]
    plt.errorbar(N_vals, means, yerr=stds, label=f'C={C}', marker='o')
plt.axhline(bayes_error, color='k', linestyle='--', label=f'Bayes eps*={bayes_error:.4f}')
plt.xlabel('sample size per class (n)')
plt.ylabel('expected test error rate')
plt.title(f'Expected test error vs n (R={R})')
plt.legend()
plt.grid(True)
plt.savefig('c06_svm_error_vs_n.png', bbox_inches='tight', facecolor='white')
plt.close()

# error vs C for each n (C axis log-scaled)
plt.figure(figsize=(8, 6), dpi=150)
for n in N_vals:
    means = [next(item['mean_error'] for item in agg_results if item['n'] == n and item['C'] == C) for C in C_list]
    stds = [next(item['std_error'] for item in agg_results if item['n'] == n and item['C'] == C) for C in C_list]
    plt.errorbar(C_list, means, yerr=stds, label=f'n={n}', marker='o')
plt.xscale('log')
plt.axhline(bayes_error, color='k', linestyle='--', label=f'Bayes eps*={bayes_error:.4f}')
plt.xlabel('C (log scale)')
plt.ylabel('expected test error rate')
plt.title(f'Expected test error vs C (R={R})')
plt.legend()
plt.grid(True)
plt.savefig('c06_svm_error_vs_C.png', bbox_inches='tight', facecolor='white')
plt.close()

# Print top-5 by mean error
agg_sorted = sorted(agg_results, key=lambda x: x['mean_error'])
print('\nBayes error (eps*):', bayes_error)
print('\nTop 5 classifier combos by mean test error (lowest first):')
print(' rank | n   |    C   | mean_err | std_err | fname')
for i, r in enumerate(agg_sorted[:5], start=1):
    print(f" {i:>3}  | {r['n']:>3} | {r['C']:>6} | {r['mean_error']:.4f} | {r['std_error']:.4f} | {r['fname']}")

print(f"\nAll results saved to {out_csv}. Generated {len(agg_results)} classifiers and {len(agg_results)} plots.")
