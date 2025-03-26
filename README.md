#  Hyperparameter Tuning with SVM and Random Forest

This repository contains a machine learning tutorial demonstrating how to use various hyperparameter optimization techniques (Grid Search, Random Search, Halving Grid Search, and optionally Optuna) with **Support Vector Machine (SVM)** and **Random Forest** models.

---

##  Overview

The notebook walks through step-by-step examples of model training, parameter optimization, and performance visualization using:
- `scikit-learn`'s GridSearchCV, RandomizedSearchCV, and HalvingGridSearchCV
- A pipeline with scaling for clean preprocessing
- Visualization of cross-validation results using heatmaps and bar charts

---

##  Dataset

- **Name:** Breast Cancer Wisconsin Diagnostic Dataset
- **Source:** Built-in via `sklearn.datasets.load_breast_cancer`
- **Features:** 30 real-valued input features describing characteristics of cell nuclei
- **Target:** Binary classification (Malignant = 0, Benign = 1)

---

##  Techniques Covered

###  SVM + Grid Search
- Tuning hyperparameters: `C`, `gamma`, `kernel`
- 5-fold cross-validation
- Results shown in a heatmap (accuracy scores by hyperparameters)

###  Random Forest + Random Search
- Tuning: `n_estimators`, `max_depth`, `min_samples_split`
- Top 10 configurations visualized in a bar plot

### Halving Grid Search (SVM)
- Efficient exploration of parameter space using progressive resource allocation

### Optional: Optuna
- If `optuna` is installed, it shows how to automate tuning using a Bayesian approach

---

## Setup Instructions

###  Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn optuna
