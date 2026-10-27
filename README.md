# Breast-Cancer Classification: CV, Tuning & ROC-AUC Evaluation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A rigorous, leak-free machine learning pipeline for diagnostic breast cancer classification on the Wisconsin Diagnostic Dataset (569 samples &times; 30 numeric cell nuclei features).

---

## Key Highlights & Performance

- **Zero Data Leakage:** Preprocessing (StandardScaler) is encapsulated entirely inside scikit-learn Pipeline objects to ensure test data is never contaminated during cross-validation.
- **Stratified 5-Fold Cross Validation:** Preserves diagnostic class proportions across all training folds.
- **Model Comparison:** Compared Support Vector Machine (SVM) and Random Forest classifiers using GridSearchCV scored directly on oc_auc.
- **Benchmark Results on Held-out 20% Test Set (114 samples):**
  - **Tuned SVM:** **98.25% Accuracy**, **98.61% Recall** (only 1 missed malignant case), **0.998 ROC-AUC**.
  - **Random Forest:** **0.993 ROC-AUC**.

| Model | CV Mean ROC-AUC | Test ROC-AUC | Test Accuracy | Test Recall (Sensitivity) | Test F1-Score |
|---|---|---|---|---|---|
| **Tuned SVM (RBF / Scaled)** | **0.994** | **0.998** | **98.3%** | **98.6%** | **0.986** |
| Random Forest (Ensemble) | 0.989 | 0.993 | 96.5% | 97.2% | 0.972 |

---

## Project Structure

`
├── model_evaluation.ipynb   # Executed Jupyter notebook with EDA, ROC curves & confusion matrices
├── run_evaluation.py        # Standalone Python pipeline & evaluation execution script
├── evaluation_metrics.json  # Machine-readable evaluation report
└── README.md                # Project documentation and reproduction guide
`

---

## Quickstart

`ash
# Clone the repository
git clone https://github.com/Jayasri9717/breast-cancer-ml.git
cd breast-cancer-ml

# Install requirements
pip install scikit-learn pandas numpy matplotlib

# Run pipeline evaluation
python run_evaluation.py
`

---

## Author

**Jayasri R**  
- Portfolio: [https://jayasri9717.github.io/jayasri-portofolio-/](https://jayasri9717.github.io/jayasri-portofolio-/)  
- GitHub: [https://github.com/Jayasri9717](https://github.com/Jayasri9717)  
- LinkedIn: [https://linkedin.com/in/jayasri-r-348baa384](https://linkedin.com/in/jayasri-r-348baa384)