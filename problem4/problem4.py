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
    'pearlite+spheroidite',
    'pearlite+widmanstatten',
    'martensite'
]

def read_micrograph_csv(path=MICRO_CSV):
    df = pd.read_csv(path)
    df = df[['path', 'primary_microconstituent']].dropna()
    return df

def collect_files_by_label(df, include_others=False):
    mapping = defaultdict(list)
    for _, row in df.iterrows():
        fname = str(row['path']).strip()
        label = str(row['primary_microconstituent']).strip().lower()
        if label in LABELS:
            mapping[label].append(fname)
        elif include_others and label in OTHER_LABELS:
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
    preds = apply_multilabel_voting(pair_results, all_files, feats, test_files_all)
    test_err = np.mean([preds[i] != test_labels_all[i] for i in range(len(preds))])
    return preds, test_err

def apply_pairwise_classifier(pair_results, label_a, label_b, all_files, feats, test_files):
    """Apply a specific pairwise classifier to test files"""
    pair_key = (label_a, label_b)
    if pair_key not in pair_results:
        # Try reversed key
        pair_key = (label_b, label_a)
        reversed_labels = True
    else:
        reversed_labels = False
    
    if pair_key not in pair_results:
        return [None] * len(test_files)
    
    info = pair_results[pair_key]
    lname = info['best_layer']
    clf = info['clf']
    
    preds = []
    for f in test_files:
        try:
            Xf = get_feature_matrix_for_files([f], all_files, feats, lname)
            p = clf.predict(Xf)[0]
            if reversed_labels:
                # If we used reversed key, flip the prediction
                pred_label = label_b if p == 0 else label_a
            else:
                pred_label = label_a if p == 0 else label_b
            preds.append(pred_label)
        except (ValueError, IndexError, KeyError):
            # File not in features or prediction failed
            preds.append(None)
    
    return preds

def apply_multilabel_voting(pair_results, all_files, feats, test_files):
    """Apply multilabel voting classifier to test files"""
    pair_keys = list(pair_results.keys())
    preds = []
    for f in test_files:
        votes = []
        for (a, b) in pair_keys:
            info = pair_results[(a, b)]
            lname = info['best_layer']
            clf = info['clf']
            try:
                Xf = get_feature_matrix_for_files([f], all_files, feats, lname)
                p = clf.predict(Xf)[0]
                voted_label = a if p == 0 else b
                votes.append(voted_label)
            except (ValueError, IndexError, KeyError):
                # File not in features or prediction failed, skip this vote
                continue
        vote_counts = Counter(votes)
        if len(vote_counts) == 0:
            final = None
        else:
            final = vote_counts.most_common(1)[0][0]
        preds.append(final)
    return preds


def main():
    os.chdir(os.path.dirname(__file__) or '.')
    print('Reading micrograph CSV...')
    df = read_micrograph_csv(MICRO_CSV)
    mapping = collect_files_by_label(df, include_others=True)
    for lab in LABELS:
        print(lab, 'count', len(mapping.get(lab, [])))

    feats_npz = 'features.npz'
    if not os.path.exists(feats_npz):
        raise FileNotFoundError(f'Features file {feats_npz} not found.')
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

    # Part (c): Apply classifiers on mixed pearlite+spheroidite micrographs
    print('\n' + '='*80)
    print('Part (c): Mixed pearlite+spheroidite test micrographs')
    print('='*80)
    
    # The label in the dataset is 'pearlite+spheroidite' (not 'mixed pearlite+spheroidite')
    mixed_ps_files = mapping.get('pearlite+spheroidite', [])
    
    if mixed_ps_files:
        # Apply pairwise pearlite vs. spheroidite classifier
        pairwise_preds = apply_pairwise_classifier(
            pair_results, 'pearlite', 'spheroidite', 
            all_files, feats, mixed_ps_files
        )
        
        # Apply multilabel voting classifier
        multilabel_preds = apply_multilabel_voting(
            pair_results, all_files, feats, mixed_ps_files
        )
        
        # Print results side by side
        print(f'\n{"Micrograph":<25} {"Pairwise (P vs S)":<20} {"Multilabel Voting":<20}')
        print('-' * 65)
        for i, fname in enumerate(mixed_ps_files):
            print(f'{fname:<25} {str(pairwise_preds[i]):<20} {str(multilabel_preds[i]):<20}')
        
        print('\nComments:')
        print('- The pairwise classifier (pearlite vs. spheroidite) is forced to choose')
        print('  between only two classes, even though the micrographs contain both.')
        print('- The multilabel voting classifier considers all 4 trained classes and')
        print('  makes predictions based on voting across all pairwise classifiers.')
        print(f'- Total micrographs evaluated: {len(mixed_ps_files)}')
    else:
        print('No mixed pearlite+spheroidite micrographs found in dataset.')

    # Part (d): Apply multilabel classifier on pearlite+widmanstatten and martensite
    print('\n' + '='*80)
    print('Part (d): Pearlite+Widmanstatten and Martensite micrographs')
    print('='*80)
    
    pw_files = mapping.get('pearlite+widmanstatten', [])
    martensite_files = mapping.get('martensite', [])
    
    print('\n--- Pearlite+Widmanstatten micrographs ---')
    if pw_files:
        pw_preds = apply_multilabel_voting(pair_results, all_files, feats, pw_files)
        print(f'\n{"Micrograph":<25} {"Multilabel Prediction":<20}')
        print('-' * 45)
        for i, fname in enumerate(pw_files):
            print(f'{fname:<25} {str(pw_preds[i]):<20}')
        print(f'\nTotal micrographs: {len(pw_files)}')
    else:
        print('No pearlite+widmanstatten micrographs found in dataset.')
    
    print('\n--- Martensite micrographs ---')
    if martensite_files:
        martensite_preds = apply_multilabel_voting(pair_results, all_files, feats, martensite_files)
        print(f'\n{"Micrograph":<25} {"Multilabel Prediction":<20}')
        print('-' * 45)
        for i, fname in enumerate(martensite_files):
            print(f'{fname:<25} {str(martensite_preds[i]):<20}')
        print(f'\nTotal micrographs: {len(martensite_files)}')
    else:
        print('No martensite micrographs found in dataset.')
    
    print('\nComparison with Part (c):')
    print('- Part (c) dealt with mixed pearlite+spheroidite, which contains two of the')
    print('  trained classes (pearlite and spheroidite).')
    print('- Part (d) deals with pearlite+widmanstatten (contains pearlite, a trained class)')
    print('  and martensite (not in any of the 4 trained classes).')
    print('- The classifier can only predict one of the 4 trained labels:')
    print('  spheroidite, network, pearlite, or spheroidite+widmanstatten.')
    print('- For martensite, the classifier will misclassify since martensite was never')
    print('  seen during training.')
    print('- For pearlite+widmanstatten, predictions may lean toward pearlite or other')
    print('  classes depending on visual similarity.')


if __name__ == '__main__':
    main()
