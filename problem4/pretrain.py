import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tqdm import tqdm
import warnings

# --- Configuration ---

IMAGE_DIR = './CMU-UHCS_Dataset/images/' 

CSV_FILE = './CMU-UHCS_Dataset/micrograph.csv'
OUTPUT_FILE = 'uhcs_vgg16_features.npz'

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. Load VGG16 Model (once) ---
base_model = VGG16(weights='imagenet', include_top=False)

# Create a model that outputs all 5 pooling layers
layer_names = ['block1_pool', 'block2_pool', 'block3_pool', 'block4_pool', 'block5_pool']
models = {lname: Model(inputs=base_model.input, outputs=base_model.get_layer(lname).output) for lname in layer_names}

print("Model loaded.")

# --- 2. Load CSV and Find Images ---
print(f"Reading image list from {CSV_FILE}...")
try:
    df = pd.read_csv(CSV_FILE)
    # We only care about the file path and the primary label
    image_list = df[['path', 'primary_microconstituent']].values
except FileNotFoundError:
    print(f"Error: {CSV_FILE} not found. Please place it in the same directory.")
    exit()

all_features = {}

# --- 3. Loop, Preprocess, and Featurize ---
print(f"Starting feature extraction for {len(image_list)} images...")
print(f"Images will be loaded from: {IMAGE_DIR}")

# Use tqdm for a progress bar
for img_filename, label in tqdm(image_list):
    img_path = os.path.join(IMAGE_DIR, img_filename)

    if not os.path.exists(img_path):
        print(f"\nWarning: Image not found, skipping: {img_path}")
        continue
    
    try:
        # 1. Load and crop image
        img = image.load_img(img_path)
        x = image.img_to_array(img)
        x = x[0:484, :, :]  # Crop subtitles

        # 2. Preprocess for VGG16
        x = np.expand_dims(x, axis=0) # Add batch dimension
        x = preprocess_input(x)       # VGG16 preprocessing

        # 3. Get all 5 feature maps in one prediction
        feature_maps_list = [models[lname].predict(x, verbose=0) for lname in layer_names]
        
        # 4. Calculate the mean for each feature map
        # This results in 5 feature vectors of lengths 64, 128, 256, 512, 512
        features_mean = [np.mean(f_map, axis=(0, 1, 2)) for f_map in feature_maps_list]
        
        # 5. Store the results in a dictionary
        all_features[img_filename] = {
            'label': label,
            'block1_mean': features_mean[0],
            'block2_mean': features_mean[1],
            'block3_mean': features_mean[2],
            'block4_mean': features_mean[3],
            'block5_mean': features_mean[4]
        }
        
    except Exception as e:
        print(f"\nError processing {img_path}: {e}")

# --- 4. Save Features to Disk ---
print(f"\nProcessed {len(all_features)} images.")
print(f"Saving features to {OUTPUT_FILE}...")
np.savez_compressed(OUTPUT_FILE, features=all_features)

print("--- Feature extraction complete! ---")

