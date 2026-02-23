# Breathy Speech Classification: GFCCs vs. MFCCs using LSTM 

This repository provides a framework for classifying phonation types (specifically Breathy vs. Modal voice) using acoustic features and deep learning. It evaluates the effectiveness of GFCC (Gammatone Frequency Cepstral Coefficients) compared to MFCC using an LSTM network.

## Key Features
* **Auditory-based Feature Extraction**: MATLAB scripts to extract 33-dimensional GFCCs (Auditory-based) and MFCCs.

* **Attention-based LSTM**: A sequence-to-one model that identifies salient phonation segments using an Attention mechanism.

* **Evaluation**: Performance assessment via Stratified 5-Fold Cross-Validation, ensuring generalizability across different speakers and languages.

## Directory Structure
<pre>
.
├── src/
│   ├── feature_extracter/      # MATLAB scripts for GFCC/MFCC extraction
│   ├── ML/        # Data loader, training, and evaluation scripts for a classification task using LSTM
├── data/
│   └── annotations/   # Metadata and phonation labels (CSV)
├──results
├── requirements.txt   # Python dependency list
└── README.md          # Project documentation
</pre>

## Prerequisites
* Python Environment
* Python 3.12+
* PyTorch >= 2.0.0
Scikit-learn, Pandas, Numpy, tqdm, TensorBoard


## MATLAB Environment
* MATLAB (R2020b or later recommended)
* Audio Toolbox
* Auditory Modeling Toolbox (AMT): Required for Gammatone filterbank processing.


## AMT Installation
Download the "AMT full package" from the https://amtoolbox.org.

Unzip the package and open MATLAB.

Navigate to the AMT directory in MATLAB and run the following command for the first-time setup:

Matlab
```bash
amt_start('install');
```
This command checks for required toolboxes, modifies the search paths, and compiles binary files.


# Usage
1. **Feature Extraction (MATLAB)**
Extract features from .wav files into JSON format using the provided MATLAB scripts:

* Extract MFCCs: 
```bash
Run src/feature_extracter/MFCC_extracter.m.
```
* Extract GFCCs: 
```bash
Run src/feature_extracter/GFCC_Extracter.m.
```

2. **Training**
Perform a grid search to find the optimal LSTM parameters. The script saves the best weights for each trial automatically.
```bash
python src/ML/Train.py [EXP_ID] --feature_type GFCC
```
3. Evaluation (Python)
Run Stratified 5-Fold Cross-Validation on the unified dataset:

```bash
python src/ML/5fold_cross.py [EXP_ID] --feature_type GFCC
```
This will output a master_log_5fold.csv containing metrics for each fold.
