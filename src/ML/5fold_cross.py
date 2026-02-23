"""
Stratified 5-Fold Cross-Validation (Single Directory Version)
------------------------------------------------------------
This script performs 5-Fold Cross-Validation on a unified dataset where all 
audio features (for all languages) are stored in a single directory.

The script uses StratifiedKFold to maintain the distribution of phonation 
labels (e.g., Breathy vs. Others) across each fold.

Author: Ikumi Saito
Date: 2026.2.20
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import pandas as pd
import os
import sys
import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix

# Imports from Train2.py (Core architecture and utilities)
from Dataset_Loader import PhonationDataset 
from Train2 import LSTMAttention, EarlyStopping, collate_fn, set_seed

# =================================================================
# 1. PATH & PARAMETER CONFIGURATION
# =================================================================
# Directory containing all JSON feature files for all languages
JSON_DIR         = ''
# Master CSV containing filenames and labels for all data
ANNOTATIONS_FILE = ''
# Results output directory
BASE_RESULTS_DIR = ''

# Optimized hyperparameters (as identified during Grid Search)
FIXED_PARAMS = {
    'hidden_size': 256,       
    'num_layers': 2,          
    'learning_rate': 0.001,   
    'batch_size': 10,         
    'dropout_rate': 0.4       
}

INPUT_SIZE = 33 
NUM_EPOCHS = 10   
PATIENCE   = 7

# =================================================================
# 2. EXPERIMENT LOGIC
# =================================================================

def run_5fold_experiment(feature_type, exp_num):
    """Main execution loop for 5-Fold Cross-Validation on a single pool."""
    set_seed(42) # Ensure reproducibility
    
    # --- Data Loading ---
    print(f"Loading unified dataset from: {JSON_DIR}")
    # Load all data using the custom Dataset class
    full_dataset = PhonationDataset(JSON_DIR, ANNOTATIONS_FILE, feature_type, INPUT_SIZE)
    
    # Extract labels from the cache for stratification purposes
    all_labels = np.array([l.item() for l in full_dataset.labels_cache])
    print(f"Total samples loaded: {len(full_dataset)}")
    
    # Create experiment directories
    main_exp_id = f'exp_5fold_{feature_type}_{exp_num:03d}'
    main_exp_dir = os.path.join(BASE_RESULTS_DIR, main_exp_id)
    os.makedirs(main_exp_dir, exist_ok=True)
    master_log_path = os.path.join(main_exp_dir, 'master_log_5fold.csv')

    # --- Stratified 5-Fold Initialization ---
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for fold, (train_val_idx, test_idx) in enumerate(skf.split(np.zeros(len(all_labels)), all_labels)):
        print(f"\n>>> Starting Fold {fold+1}/5")
        fold_dir = os.path.join(main_exp_dir, f'fold_{fold+1}')
        os.makedirs(fold_dir, exist_ok=True)

        # Stratified split: 80% Train+Val, 20% Test
        train_val_labels = all_labels[train_val_idx]
        
        # Further split Train+Val into Train (80%) and Val (20%) for Early Stopping
        train_idx_rel, val_idx_rel = train_test_split(
            np.arange(len(train_val_idx)), 
            test_size=0.2, 
            stratify=train_val_labels, 
            random_state=42
        )
        
        train_subset = Subset(full_dataset, train_val_idx[train_idx_rel])
        val_subset   = Subset(full_dataset, train_val_idx[val_idx_rel])
        test_subset  = Subset(full_dataset, test_idx)

        # DataLoaders using custom collate and sampler logic from Train2.py
        train_loader = DataLoader(train_subset, batch_size=FIXED_PARAMS['batch_size'], shuffle=True, collate_fn=collate_fn)
        val_loader   = DataLoader(val_subset, batch_size=FIXED_PARAMS['batch_size'], shuffle=False, collate_fn=collate_fn)
        test_loader  = DataLoader(test_subset, batch_size=FIXED_PARAMS['batch_size'], shuffle=False, collate_fn=collate_fn)

        # --- Model Initialization ---
        # Initialize LSTM with Attention architecture
        model = LSTMAttention(INPUT_SIZE, FIXED_PARAMS['hidden_size'], FIXED_PARAMS['num_layers'], 2, FIXED_PARAMS['dropout_rate']).to(device)
        optimizer = optim.Adam(model.parameters(), lr=FIXED_PARAMS['learning_rate'])
        criterion = nn.BCEWithLogitsLoss()
        early_stopping = EarlyStopping(patience=PATIENCE)
        
        history_log = []
        best_val_acc = 0

        # --- Training Loop ---
        for epoch in range(NUM_EPOCHS):
            model.train()
            t_loss, t_corr, t_total = 0, 0, 0
            for feat, lbl, lens in train_loader:
                feat, lbl, lens = feat.to(device), lbl.to(device).float(), lens.to(device)
                optimizer.zero_grad()
                outputs = model(feat, lens)
                loss = criterion(outputs, lbl)
                loss.backward(); optimizer.step()
                t_loss += loss.item()
                t_corr += ((torch.sigmoid(outputs) > 0.5).float() == lbl).sum().item()
                t_total += lbl.size(0)

            # --- Validation Phase ---
            model.eval()
            v_loss_sum, v_corr, v_total = 0, 0, 0
            with torch.no_grad():
                for feat, lbl, lens in val_loader:
                    feat, lbl, lens = feat.to(device), lbl.to(device).float(), lens.to(device)
                    outputs = model(feat, lens)
                    v_loss_sum += criterion(outputs, lbl).item()
                    v_corr += ((torch.sigmoid(outputs) > 0.5).float() == lbl).sum().item()
                    v_total += lbl.size(0)
            
            avg_val_acc = 100 * v_corr / v_total
            avg_val_loss = v_loss_sum / len(val_loader)
            
            history_log.append({
                'epoch': epoch + 1, 
                'train_acc': 100 * t_corr / t_total, 
                'val_acc': avg_val_acc,
                'val_loss': avg_val_loss
            })
            print(f"Fold {fold+1} Epoch {epoch+1} | Train Acc: {history_log[-1]['train_acc']:.2f}% | Val Acc: {avg_val_acc:.2f}%")

            # Save the best model weights for the current fold
            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                torch.save(model.state_dict(), os.path.join(fold_dir, 'best_model.pth'))

            early_stopping(avg_val_loss)
            if early_stopping.early_stop: break

        pd.DataFrame(history_log).to_csv(os.path.join(fold_dir, 'learning_history.csv'), index=False)

        # --- Final Fold Evaluation ---
        # Load the weights that performed best on the validation set
        model.load_state_dict(torch.load(os.path.join(fold_dir, 'best_model.pth'), weights_only=True))
        model.eval()
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for feat, lbl, lens in test_loader:
                feat, lbl, lens = feat.to(device), lbl.to(device).float(), lens.to(device)
                outputs = model(feat, lens)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(lbl.cpu().numpy())

        # Performance metrics using Confusion Matrix
        cm = confusion_matrix(all_targets, all_preds)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate balanced accuracy and specificities
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        balanced_acc = (recall + specificity) / 2
        accuracy = (tp + tn) / len(all_targets)

        print(f"Fold {fold+1} Result -> Balanced Acc: {balanced_acc:.4f}, Recall: {recall:.4f}")

        # Log fold summary to master CSV
        log_data = FIXED_PARAMS.copy()
        log_data.update({
            'fold': fold + 1,
            'balanced_accuracy': balanced_acc,
            'accuracy': accuracy,
            'recall': recall,
            'specificity': specificity,
            'TP': tp, # True Positives
            'TN': tn, # True Negatives
            'FP': fp, # False Positives
            'FN': fn, # False Negatives
            'feature_type': feature_type
        })
        pd.DataFrame([log_data]).to_csv(master_log_path, mode='a', header=not os.path.exists(master_log_path), index=False)

    # --- Final Cross-Validation Summary ---
    final_df = pd.read_csv(master_log_path)
    print("\n" + "="*55)
    print(f" 5-FOLD CROSS VALIDATION FINAL SUMMARY ({feature_type})")
    print(f" Mean Balanced Accuracy: {final_df['balanced_accuracy'].mean():.4f}")
    print(f" Mean Recall (Breathy):  {final_df['recall'].mean():.4f}")
    print(f" Mean Specificity:       {final_df['specificity'].mean():.4f}")
    print("="*55)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-language 5-Fold Cross-Validation.")
    parser.add_argument("exp_num", type=int, help="Experiment sequence number.")
    parser.add_argument("--feature_type", type=str, default="GFCC", choices=["MFCC", "GFCC"])
    args = parser.parse_args()
    
    run_5fold_experiment(args.feature_type, args.exp_num)