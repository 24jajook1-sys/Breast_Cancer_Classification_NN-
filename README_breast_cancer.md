# Breast Cancer Classification using Neural Networks

Classifying breast tumors as **malignant** or **benign** on the Wisconsin Breast Cancer Diagnostic dataset — comparing three classical ML models against a deep neural network, with full EDA, feature engineering, and evaluation.

## Overview

This project walks through a complete supervised learning pipeline for a binary medical classification problem:

- Exploratory data analysis (distributions, correlations, outliers)
- Feature engineering with PCA and mutual information
- Four models trained and cross-validated: Logistic Regression, SVM (RBF), Random Forest, and a Keras neural network
- Full evaluation suite: accuracy, precision, recall, F1, ROC/AUC, PR curves, confusion matrices
- A saved, reloadable prediction system with multi-model consensus

## Dataset

**Wisconsin Breast Cancer Diagnostic** dataset (via `sklearn.datasets.load_breast_cancer`)
- 569 samples, 30 numeric features (computed from digitized images of a fine needle aspirate of a breast mass)
- Binary target: malignant (0) / benign (1)

## Project workflow

| Step | Section |
|------|---------|
| 1 | Setup & imports |
| 2 | Data loading |
| 3 | EDA — basic statistics |
| 4 | EDA — advanced visualization (heatmap, boxplots, violin plots, pairplot) |
| 5 | Feature engineering (mutual information, PCA 2D/3D, scree plot) |
| 6 | Data preprocessing (stratified split, standardization) |
| 7–9 | Logistic Regression / SVM / Random Forest (with 5-fold CV) |
| 10 | Multi-model comparison |
| 11–12 | Neural network — build & train |
| 13 | Advanced evaluation across all 4 models |
| 14 | Predictive system (multi-model consensus) |
| 15 | Save & reload models |

## Models compared

| Model | Type |
|---|---|
| Logistic Regression | Linear baseline |
| SVM (RBF kernel) | Non-linear classical ML |
| Random Forest | Ensemble, with feature importances |
| Neural Network (Keras) | Dense(256)→Dense(128)→Dense(64)→Dense(32)→Dense(2), with BatchNorm + Dropout, EarlyStopping and ReduceLROnPlateau |

Exact accuracy/precision/recall/F1/ROC-AUC for each model are printed and plotted when the notebook is run — see Step 13.

## Tech stack

Python · NumPy · pandas · scikit-learn · TensorFlow / Keras · Matplotlib · Seaborn

## Project structure

```
├── Breast_Cancer_Classification_NN.ipynb   # Main notebook (all 15 steps)
├── breast_cancer_nn_model.keras            # Saved neural network (generated on run)
├── scaler.pkl                              # Saved StandardScaler (generated on run)
├── sklearn_models.pkl                      # Saved LogReg/SVM/RF models (generated on run)
└── README.md
```

## Getting started

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn
jupyter notebook Breast_Cancer_Classification_NN.ipynb
```

Run all cells top to bottom — the dataset loads directly from scikit-learn, so no external download is needed.

## Author

Kush Shreeram Jajoo
