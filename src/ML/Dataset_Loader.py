"""
Phonation Dataset Loader with Global Normalization
This script defines a PyTorch Dataset class for loading speech features (MFCC/GFCC)
and applying dataset-wide normalization.

Author: [Ikumi Saito]
Date: 2026.2.20
"""

import torch
from torch.utils.data import Dataset
import json
import pandas as pd
import numpy as np
import os
from tqdm import tqdm 

class PhonationDataset(Dataset):
    """
    Dataset class for Phonation Type Classification.
    Loads audio features from JSON files and applies global mean-variance normalization.
    """
    
    def __init__(self, json_dir, annotations_file, feature_type='gfcc', num_features=33): 
        self.feature_type = feature_type.lower()
        self.num_features = num_features 
        self.label_column_name = 'phonation_label'
        
        # Mapping: 'B' (Breathy) as positive class, others as negative
        self.binary_label_map = {'B': 1, 'M': 0, 'C': 0, 'T': 0, 'A': 0, 'AA': 0, 'UA': 0, 'G': 0}

        all_annotations = pd.read_csv(annotations_file)
        self.data_cache = [] 
        self.labels_cache = [] 
        
        print(f"--- Loading {self.feature_type.upper()} features into RAM ---")

        # 1. Raw Data Loading (Pre-normalization)
        for index, row in tqdm(all_annotations.iterrows(), total=len(all_annotations)):
            original_filename = row.iloc[0] 
            json_filename = original_filename.replace('.wav', '.json')
            json_path = os.path.join(json_dir, json_filename)
            
            if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    
                    # Sort and select features based on the defined type and dimension
                    feature_keys = sorted(
                       [key for key in data.keys() if key.startswith(f'{self.feature_type}_')],
                       key=lambda x: int(x.split('_')[-1]) 
                    )[:self.num_features]
                    
                    if not feature_keys:
                        continue 

                    # Stack features into a [Time, Coefficients] matrix
                    feature_list = [np.atleast_1d(data[key]) for key in feature_keys]
                    feature_matrix = np.stack(feature_list, axis=1)
                    
                    features_tensor = torch.tensor(feature_matrix, dtype=torch.float32)
                    
                    # Assign label based on mapping
                    label_name = row[self.label_column_name]
                    numeric_label = self.binary_label_map.get(label_name, 0)
                    label_tensor = torch.tensor(numeric_label, dtype=torch.long)
                    
                    self.data_cache.append(features_tensor)
                    self.labels_cache.append(label_tensor)  
                except Exception as e:
                    print(f"\n[Warning] Error loading {json_filename}: {e}")
            
        print(f"Found {len(self.data_cache)} valid files.")
        
        # 2. Global Normalization Calculation
        # Instead of normalizing each file independently, we normalize based on the whole dataset statistics.
        # if len(self.data_cache) > 0:
            # print("Calculating Global Mean and Std...")
            
            # Concatenate all frames across all files to compute population statistics
            # all_features_concat = torch.cat(self.data_cache, dim=0) 
            
            # Compute stats along the feature dimension
            # global_mean = torch.mean(all_features_concat, dim=0)
            # global_std = torch.std(all_features_concat, dim=0) + 1e-8 
            
            # print(f"Global Mean (first 5): {global_mean[:5]}")
           # print(f"Global Std  (first 5): {global_std[:5]}")

           # print("Applying Global Normalization...")
          #  for i in range(len(self.data_cache)):
              #  self.data_cache[i] = (self.data_cache[i] - global_mean) / global_std
            
           # print("Normalization complete.")
    def __len__(self):
        return len(self.data_cache)

    def __getitem__(self, index):
        return self.data_cache[index], self.labels_cache[index]