# data_processing.py
import cv2 #pour traiter les images
import os #pour les opérations liées au systéme de fichiers
import numpy as np
from descriptor import glcm, bitdesc , glcm_bit

def extract_features(image_path, descriptor_func):
    if not isinstance(image_path, str):
        print(f"Invalid image path type: {type(image_path)}. Expected str.")
        return None
    
    if not os.path.isfile(image_path):
        return None
    
    img = cv2.imread(image_path, 0)
    if img is None:
        return None
    
    print(f"Image type: {type(img)}, Image shape: {img.shape}")  # Debugging line
    features = descriptor_func(img)
    return features

def process_datasets(root_folder, descriptor_func, output_file):
    all_features = [] 
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_rel_path = os.path.join(root, file)
                print(f"Processing file: {image_rel_path}")  # Debugging line
                
                if not os.path.isfile(image_rel_path):
                    print(f"File does not exist: {image_rel_path}")  # Debugging line
                    continue
                
                relative_path = os.path.relpath(image_rel_path, root_folder)
                folder_name = os.path.basename(os.path.dirname(image_rel_path))
                
                # Extract features
                features = extract_features(image_rel_path, descriptor_func)
                if features is not None:
                    features = features + [folder_name, relative_path]
                    all_features.append(features)
    
    print(f"Extracted features: {all_features}")
    signatures = np.array(all_features, dtype=object)
    np.save(output_file, signatures)
    print(f'Successfully stored in {output_file}!')

process_datasets('./dataset', glcm, 'glcm_signatures.npy')

process_datasets('./dataset', bitdesc, 'bitdesc_signatures.npy')

process_datasets('./dataset', glcm_bit, 'glcm_bit_signatures.npy')

