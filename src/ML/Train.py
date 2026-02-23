"""
LSTM with Attention for Phonation Classification
-----------------------------------------------
This script performs a hyperparameter grid search. 
Model weights are saved automatically for EACH trial when a new best 
validation accuracy is reached. Redundant final training is removed.

Author: [Ikumi Saito]
Date: 2026.2.20
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, BatchSampler
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.tensorboard import SummaryWriter

import pandas as pd
import numpy as np
import os
import sys
import json
import random
import itertools
import argparse
import shutil
from datetime import datetime

# Import the custom dataset class
from Dataset_Loader import PhonationDataset 

# =================================================================
# --- Path Configuration ---
# =================================================================
JSON_DIR         = ''
ANNOTATIONS_FILE = ''
BASE_RESULTS_DIR = ''


# --- Training Constants ---
INPUT_SIZE        = 33 
NUM_CLASSES       = 2  
NUM_EPOCH = 50 

# --- HyperParameter Settings ---
GRID_SEARCH_SPACE = {
    'hidden_size': [256],
    'num_layers': [2],
    'learning_rate': [0.001],
    'batch_size': [10],
    'dropout_rate': [0.4]
}

# =================================================================
# UTILITIES
# =================================================================

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EarlyStopping:
    def __init__(self, patience=7, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

class LengthSortedBatchSampler(BatchSampler): 
    def __init__(self, subset, batch_size, shuffle=True):
        self.subset = subset
        self.batch_size = batch_size
        self.shuffle = shuffle
        indices_with_lengths = []
        full_data_cache = subset.dataset.data_cache 
        for relative_idx, original_idx in enumerate(subset.indices):
            length = len(full_data_cache[original_idx]) 
            indices_with_lengths.append((relative_idx, length))
        indices_with_lengths.sort(key=lambda x: x[1]) 
        self.sorted_indices = [item[0] for item in indices_with_lengths]
        
    def __iter__(self):
        batches = [self.sorted_indices[i:i + self.batch_size] for i in range(0, len(self.sorted_indices), self.batch_size)]
        if self.shuffle: random.shuffle(batches) 
        for batch in batches: yield batch 

    def __len__(self):
        return (len(self.sorted_indices) + self.batch_size - 1) // self.batch_size

# =================================================================
# MODEL DEFINITION
# =================================================================

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.query_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden, encoder_outputs, lengths):
        query = self.query_layer(hidden).unsqueeze(1) 
        keys = self.key_layer(encoder_outputs)
        energy = torch.tanh(query + keys)
        attn_weights = torch.sum(self.v * energy, dim=2)
        mask = torch.arange(encoder_outputs.size(1), device=lengths.device)[None, :] < lengths[:, None]
        attn_weights.masked_fill_(~mask, float('-inf'))
        return F.softmax(attn_weights, dim=1)

class LSTMAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate):
        super(LSTMAttention, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.attention = Attention(hidden_size)
        self.dropout_fc = nn.Dropout(dropout_rate) 
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, lengths):
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, (hidden, _) = self.lstm(packed_input)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        attn_weights = self.attention(hidden[-1], lstm_out, lengths)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        return self.fc(self.dropout_fc(context)).squeeze(1)

def collate_fn(batch):
    features, labels = zip(*batch)
    lengths = torch.tensor([len(f) for f in features])
    features_padded = pad_sequence(features, batch_first=True, padding_value=0)
    return features_padded, torch.stack(labels), lengths

# =================================================================
# TRIAL LOGIC
# =================================================================

def run_grid_trial(params, train_dataset, val_dataset, results_dir, trial_number):
    """Executes a single trial and SAVES the best model for this trial to disk."""
    writer = SummaryWriter(log_dir=os.path.join(results_dir, 'tensorboard_logs', f'trial_{trial_number:03d}'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = LSTMAttention(INPUT_SIZE, params['hidden_size'], params['num_layers'], NUM_CLASSES, params['dropout_rate']).to(device)
    if torch.cuda.device_count() > 1: model = nn.DataParallel(model)

    train_loader = DataLoader(train_dataset, batch_sampler=LengthSortedBatchSampler(train_dataset, params['batch_size']),
                              collate_fn=collate_fn, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_sampler=LengthSortedBatchSampler(val_dataset, params['batch_size'], shuffle=False),
                            collate_fn=collate_fn, num_workers=16, pin_memory=True)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    early_stopping = EarlyStopping(patience=7)

    best_val_acc = 0.0
    best_epoch = 0
    trial_model_path = os.path.join(results_dir, f'trial_{trial_number:03d}_best.pth')

    for epoch in range(NUM_EPOCH):
        # Training Phase
        model.train()
        train_correct, train_total, train_loss = 0, 0, 0.0
        for feat, lbl, lens in train_loader:
            feat, lbl, lens = feat.to(device), lbl.to(device).float(), lens.to(device)
            optimizer.zero_grad()
            outputs = model(feat, lens)
            loss = criterion(outputs, lbl)
            loss.backward(); optimizer.step()
            
            train_loss += loss.item()
            train_correct += ((torch.sigmoid(outputs) > 0.5).float() == lbl).sum().item()
            train_total += lbl.size(0)

        # Validation Phase
        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for feat, lbl, lens in val_loader:
                feat, lbl, lens = feat.to(device), lbl.to(device).float(), lens.to(device)
                outputs = model(feat, lens)
                val_loss += criterion(outputs, lbl).item()
                val_correct += ((torch.sigmoid(outputs) > 0.5).float() == lbl).sum().item()
                val_total += lbl.size(0)
        
        val_acc = 100 * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_train_acc = 100 * train_correct / train_total
        
        print(f"  Trial {trial_number} | Epoch {epoch+1} | "
              f"Train Acc: {avg_train_acc:.2f}%, Loss: {avg_train_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}%, Loss: {avg_val_loss:.4f}")

        writer.add_scalar('Accuracy/train', avg_train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        # SAVE MODEL IF BEST FOR THIS TRIAL
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(model_to_save.state_dict(), trial_model_path)

        early_stopping(avg_val_loss)
        if early_stopping.early_stop: break

    writer.close()
    return best_val_acc, best_epoch

# =================================================================
# MAIN BLOCK
# =================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_num", type=int)
    parser.add_argument("--feature_type", type=str, default="GFCC") 
    args = parser.parse_args()
    
    set_seed(42)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_id = f'exp_{args.feature_type}_{args.exp_num:03d}'
    results_dir = os.path.join(BASE_RESULTS_DIR, experiment_id)
    os.makedirs(results_dir, exist_ok=True)
    
    # Redirection to log file
    log_path = os.path.join(results_dir, 'run.log')
    sys.stdout = open(log_path, 'w')
    sys.stderr = sys.stdout 

    print(f"--- Experiment Initialized: {experiment_id} ---")
    
    # Dataset
    full_dataset = PhonationDataset(JSON_DIR, ANNOTATIONS_FILE, args.feature_type, INPUT_SIZE)
    train_size = int(0.8 * len(full_dataset))
    train_subset, val_subset = random_split(full_dataset, [train_size, len(full_dataset) - train_size])

    # Grid Search
    keys, values = zip(*GRID_SEARCH_SPACE.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    search_results = []
    best_overall_acc = 0.0
    best_trial_index = -1

    for i, params in enumerate(combinations):
        acc, epoch = run_grid_trial(params, train_subset, val_subset, results_dir, i + 1)
        
        entry = params.copy()
        entry.update({'trial_num': i+1, 'val_accuracy': acc, 'best_epoch': epoch})
        search_results.append(entry)
        
        if acc > best_overall_acc:
            best_overall_acc = acc
            best_trial_index = i + 1
        
        pd.DataFrame(search_results).to_csv(os.path.join(results_dir, f'grid_results_{args.exp_num:03d}.csv'), index=False)

    print(f"\n--- Grid Search Finished ---")
    print(f"Best Accuracy: {best_overall_acc:.2f}% found in Trial {best_trial_index}")

    # Copy the best trial model to the final model path (instead of re-training)
    if best_trial_index != -1:
        source_path = os.path.join(results_dir, f'trial_{best_trial_index:03d}_best.pth')
        final_path = os.path.join(results_dir, f'best_model_{args.exp_num:03d}.pth')
        if os.path.exists(source_path):
            shutil.copy(source_path, final_path)
            print(f"Final model weight copied to: {final_path}")

    # Update Master Log
    master_log_path = os.path.join(BASE_RESULTS_DIR, 'master_log.csv')
    log_entry = {'exp_id': experiment_id, 'best_acc': best_overall_acc, 'best_trial': best_trial_index, 'time': timestamp}
    log_df = pd.DataFrame([log_entry])
    log_df.to_csv(master_log_path, mode='a', header=not os.path.exists(master_log_path), index=False)

    sys.stdout.close()