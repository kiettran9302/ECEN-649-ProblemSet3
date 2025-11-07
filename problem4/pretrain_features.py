#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from collections import defaultdict

try:
    # prefer tensorflow.keras
    from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.models import Model
except Exception:
    from keras.applications.vgg16 import VGG16, preprocess_input
    from keras.preprocessing import image
    from keras.models import Model

DATA_DIR = 'CMU-UHCS_Dataset'
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
MICRO_CSV = os.path.join(DATA_DIR, 'micrograph.csv')

LABELS = ['spheroidite', 'network', 'pearlite', 'spheroidite+widmanstatten']
# Include all labels for feature extraction
ALL_LABELS = LABELS + ['pearlite+spheroidite', 'martensite', 'pearlite+widmanstatten']
LAYER_NAMES = ['block1_pool', 'block2_pool', 'block3_pool', 'block4_pool', 'block5_pool']

def read_micrograph_csv(path=MICRO_CSV):
    df = pd.read_csv(path)
    df = df[['path', 'primary_microconstituent']].dropna()
    return df

def collect_files_by_label(df, labels=None):
    if labels is None:
        labels = ALL_LABELS
    mapping = defaultdict(list)
    for _, row in df.iterrows():
        fname = str(row['path']).strip()
        label = str(row['primary_microconstituent']).strip().lower()
        if label in labels:
            mapping[label].append(fname)
    return mapping

def load_and_preprocess(img_path):
    img = image.load_img(img_path)
    x = image.img_to_array(img)
    if x.shape[0] > 484:
        x = x[0:484, :, :]
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def featurize_all(mapping, save_path='features.npz'):
    base = VGG16(weights='imagenet', include_top=False)
    models = {lname: Model(inputs=base.input, outputs=base.get_layer(lname).output) for lname in LAYER_NAMES}
    features = {lname: {} for lname in LAYER_NAMES}

    for label, files in mapping.items():
        for fname in files:
            img_file = os.path.join(IMAGES_DIR, fname)
            if not os.path.exists(img_file):
                print('Warning: image not found', img_file)
                continue
            x = load_and_preprocess(img_file)
            for lname, model in models.items():
                xb = model.predict(x)
                F = np.mean(xb, axis=(0, 1, 2))
                features[lname][fname] = F.astype(np.float32)

    all_files = sorted({f for d in features.values() for f in d.keys()})
    npz_dict = {'files': np.array(all_files)}
    for lname in LAYER_NAMES:
        # determine feature dimension for layer
        if len(features[lname]) > 0:
            sample = next(iter(features[lname].values()))
            dim = sample.shape
        else:
            dim = (0,)
        arr = np.array([features[lname].get(f, np.zeros(dim, dtype=np.float32)) for f in all_files])
        npz_dict[lname] = arr

    np.savez_compressed(save_path, **npz_dict)
    print('Saved features to', save_path)
    return save_path

def main():
    import sys
    os.chdir(os.path.dirname(__file__) or '.')
    print('Reading micrograph CSV...')
    df = read_micrograph_csv(MICRO_CSV)
    mapping = collect_files_by_label(df, labels=ALL_LABELS)
    for lab in ALL_LABELS:
        print(lab, 'count', len(mapping.get(lab, [])))

    feats_npz = 'features.npz'
    force_regenerate = '--force' in sys.argv
    
    if not os.path.exists(feats_npz) or force_regenerate:
        print('Featurizing images (this may take a while)...')
        featurize_all(mapping, save_path=feats_npz)
    else:
        print('Found existing features file', feats_npz)
        print('Use --force to regenerate features')

if __name__ == '__main__':
    main()