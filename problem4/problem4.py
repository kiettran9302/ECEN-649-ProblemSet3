#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

DATA_DIR = 'CMU-UHCS_Dataset'
MICRO_CSV = os.path.join(DATA_DIR, 'micrograph.csv')

LABELS = ['spheroidite', 'network', 'pearlite', 'spheroidite+widmanstatten']
TRAIN_COUNTS = {'spheroidite': 100, 'network': 100, 'pearlite': 100, 'spheroidite+widmanstatten': 60}
LAYER_NAMES = ['block1_pool', 'block2_pool', 'block3_pool', 'block4_pool', 'block5_pool']
OTHER_LABELS = [
    'mixed pearlite+spheroidite',
    'pearlite+widmanstatten',
    'martensite'
]

def read_micrograph_csv(path=MICRO_CSV):
    df = pd.read_csv(path)
    df = df[['path', 'primary_microconstituent']].dropna()
    return df

def collect_files_by_label(df):
    mapping = defaultdict(list)
    for _, row in df.iterrows():
        fname = str(row['path']).strip()
        label = str(row['primary_microconstituent']).strip().lower()
        if label in LABELS:
            mapping[label].append(fname)

    return mapping

def load_features(npz_path='features.npz'):
    data = np.load(npz_path, allow_pickle=True)
    files = list(data['files'])
    feats = {lname: data[lname] for lname in LAYER_NAMES}
    return files, feats

def get_feature_matrix_for_files(files_list, all_files, feats, lname):
    idx = [all_files.index(f) for f in files_list]
    X = feats[lname][idx]
    return X

def train_pairwise(mapping, all_files, feats):
    from itertools import combinations
    pair_results = {}
    for a, b in combinations(LABELS, 2):
        files_a = mapping.get(a, [])
        files_b = mapping.get(b, [])
        ntrain_a = min(len(files_a), TRAIN_COUNTS.get(a, len(files_a)))
        ntrain_b = min(len(files_b), TRAIN_COUNTS.get(b, len(files_b)))
        train_files = files_a[:ntrain_a] + files_b[:ntrain_b]
        train_y = np.array([0] * ntrain_a + [1] * ntrain_b)
        test_files = files_a[ntrain_a:] + files_b[ntrain_b:]
        test_y = np.array([0] * max(0, len(files_a) - ntrain_a) + [1] * max(0, len(files_b) - ntrain_b))

        best_layer = None
        best_err = 1.0
        best_scores = None

        for lname in LAYER_NAMES:
            try:
                X = get_feature_matrix_for_files(train_files, all_files, feats, lname)
            except ValueError:
                continue
            if X.ndim != 2:
                continue
            clf = SVC(C=1, kernel='rbf', gamma='auto')
            try:
                scores = cross_val_score(clf, X, train_y, cv=10, scoring='accuracy')
            except Exception:
                continue
            err = 1.0 - scores.mean()
            if err < best_err:
                best_err = err
                best_layer = lname
                best_scores = scores

        if best_layer is None:
            print(f'Warning: no valid layer found for pair {a} vs {b}')
            continue

        X_train_best = get_feature_matrix_for_files(train_files, all_files, feats, best_layer)
        final_clf = SVC(C=1, kernel='rbf', gamma='auto')
        final_clf.fit(X_train_best, train_y)

        if len(test_files) > 0:
            X_test = get_feature_matrix_for_files(test_files, all_files, feats, best_layer)
            y_pred = final_clf.predict(X_test)
            test_err = np.mean(y_pred != test_y)
        else:
            test_err = None

        pair_results[(a, b)] = {
            'best_layer': best_layer,
            'cv_error': best_err,
            'cv_scores': best_scores,
            'clf': final_clf,
            'train_files': train_files,
            'test_files': test_files,
            'test_error': test_err,
        }
        print(f'Pair {a} vs {b}: best layer {best_layer}, CV err {best_err:.4f}, test err {test_err}')
    return pair_results

def multilabel_vote(pair_results, all_files, feats, test_files_all, test_labels_all):
    pair_keys = list(pair_results.keys())
    preds = []
    for f in test_files_all:
        votes = []
        for (a, b) in pair_keys:
            info = pair_results[(a, b)]
            lname = info['best_layer']
            clf = info['clf']
            Xf = get_feature_matrix_for_files([f], all_files, feats, lname)
            try:
                p = clf.predict(Xf)[0]
            except Exception:
                p = 0
            voted_label = a if p == 0 else b
            votes.append(voted_label)
        vote_counts = Counter(votes)
        if len(vote_counts) == 0:
            final = None
        else:
            final = vote_counts.most_common(1)[0][0]
        preds.append(final)
    test_err = np.mean([preds[i] != test_labels_all[i] for i in range(len(preds))])
    return preds, test_err


def main():
    os.chdir(os.path.dirname(__file__) or '.')
    print('Reading micrograph CSV...')
    df = read_micrograph_csv(MICRO_CSV)
    mapping = collect_files_by_label(df)
    for lab in LABELS:
        print(lab, 'count', len(mapping.get(lab, [])))

    feats_npz = 'features.npz'
    if not os.path.exists(feats_npz):
        raise FileNotFoundError(f'Features file {feats_npz} not found.1')
    all_files, feats = load_features(feats_npz)

    pair_results = train_pairwise(mapping, all_files, feats)

    test_files_all = []
    test_labels_all = []
    for lab in LABELS:
        files = mapping.get(lab, [])
        ntrain = min(len(files), TRAIN_COUNTS.get(lab, len(files)))
        test_files = files[ntrain:]
        test_files_all.extend(test_files)
        test_labels_all.extend([lab] * len(test_files))

    print('\nEvaluating multilabel voting classifier on combined 4-label test set...')
    preds, multilabel_err = multilabel_vote(pair_results, all_files, feats, test_files_all, test_labels_all)
    print('Multilabel test error:', multilabel_err)


if __name__ == '__main__':
    main()
