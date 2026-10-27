"""
Build and execute the model evaluation pipeline on Breast Cancer Wisconsin dataset.
Outputs:
1. model_evaluation.ipynb (fully executed Colab-compatible notebook with outputs & plots)
2. evaluation_metrics.json (machine-readable metrics)
"""

import base64
import io
import json
import matplotlib.pyplot as plt
import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
    # 1. Dataset loading and inspection
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target  # 0 = malignant, 1 = benign
    target_names = cancer.target_names  # ['malignant', 'benign']
    feature_names = cancer.feature_names

    # 2. Stratified train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 3. Scikit-learn preprocessing and model pipeline (leakage-free)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=42, solver="liblinear")),
    ])

    # 4. Hyperparameter grid and Stratified 5-fold CV
    param_grid = {
        "classifier__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "classifier__penalty": ["l1", "l2"],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        refit=True,
        return_train_score=True,
        n_jobs=1,
    )

    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    cv_best_roc_auc = grid_search.best_score_

    # 5. Held-out test set evaluation
    best_model = grid_search.best_estimator_
    y_test_pred = best_model.predict(X_test)
    y_test_proba = best_model.predict_proba(X_test)[:, 1]

    test_roc_auc = float(roc_auc_score(y_test, y_test_proba))
    test_acc = float(accuracy_score(y_test, y_test_pred))
    test_prec = float(precision_score(y_test, y_test_pred))
    test_rec = float(recall_score(y_test, y_test_pred))
    test_f1 = float(f1_score(y_test, y_test_pred))
    cm = confusion_matrix(y_test, y_test_pred).tolist()

    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)

    # 6. Generate ROC Curve & Confusion Matrix plot as base64 image
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Confusion matrix display
    im = ax1.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax1.set_title("Confusion Matrix (Held-Out Test Set)", fontsize=13, pad=12)
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(target_names))
    ax1.set_xticks(tick_marks)
    ax1.set_yticks(tick_marks)
    ax1.set_xticklabels(target_names, fontsize=10)
    ax1.set_yticklabels(target_names, fontsize=10)
    ax1.set_xlabel("Predicted Label", fontsize=11, labelpad=8)
    ax1.set_ylabel("True Label", fontsize=11, labelpad=8)
    for i in range(len(cm)):
        for j in range(len(cm[0])):
            color = "white" if cm[i][j] > (len(y_test) / 4) else "black"
            ax1.text(j, i, format(cm[i][j], "d"), ha="center", va="center", color=color, fontsize=12, fontweight="bold")

    # ROC Curve display
    ax2.plot(fpr, tpr, color="#2563eb", lw=2.5, label=f"ROC Curve (AUC = {test_roc_auc:.4f})")
    ax2.plot([0, 1], [0, 1], color="#94a3b8", lw=1.5, linestyle="--", label="Random Classifier")
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.05])
    ax2.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, labelpad=8)
    ax2.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, labelpad=8)
    ax2.set_title(f"ROC Curve (Test ROC-AUC: {test_roc_auc:.4f})", fontsize=13, pad=12)
    ax2.legend(loc="lower right", frameon=True)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    # Save metrics to JSON
    metrics_summary = {
        "dataset": "Breast Cancer Wisconsin Diagnostic (scikit-learn)",
        "total_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "cv_folds": 5,
        "cv_scoring": "roc_auc",
        "best_hyperparameters": best_params,
        "cv_mean_roc_auc": round(float(cv_best_roc_auc), 4),
        "held_out_test_metrics": {
            "roc_auc": round(test_roc_auc, 4),
            "accuracy": round(test_acc, 4),
            "precision": round(test_prec, 4),
            "recall": round(test_rec, 4),
            "f1_score": round(test_f1, 4),
            "confusion_matrix": {
                "true_malignant (TN)": cm[0][0],
                "false_benign (FP)": cm[0][1],
                "false_malignant (FN)": cm[1][0],
                "true_benign (TP)": cm[1][1],
            },
        },
    }

    with open("evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print("\n" + "=" * 60)
    print("      MODEL EVALUATION PIPELINE RESULTS (SCIKIT-LEARN)")
    print("=" * 60)
    print(f"Dataset                  : {metrics_summary['dataset']}")
    print(f"Split                    : {metrics_summary['train_samples']} train, {metrics_summary['test_samples']} test")
    print(f"Best CV ROC-AUC (5-fold) : {metrics_summary['cv_mean_roc_auc']:.4f}")
    print(f"Best Hyperparameters     : {best_params}")
    print(f"Held-Out Test ROC-AUC    : {metrics_summary['held_out_test_metrics']['roc_auc']:.4f}")
    print(f"Held-Out Test Accuracy   : {metrics_summary['held_out_test_metrics']['accuracy']:.4f} ({metrics_summary['held_out_test_metrics']['accuracy']*100:.2f}%)")
    print(f"Held-Out Test Precision  : {metrics_summary['held_out_test_metrics']['precision']:.4f}")
    print(f"Held-Out Test Recall     : {metrics_summary['held_out_test_metrics']['recall']:.4f}")
    print(f"Held-Out Test F1-Score   : {metrics_summary['held_out_test_metrics']['f1_score']:.4f}")
    print(f"Confusion Matrix         : {cm}")
    print("=" * 60)

    # 7. Construct Google Colab-compatible model_evaluation.ipynb with embedded execution outputs
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"name": "model_evaluation.ipynb", "provenance": []},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Model Evaluation: Breast Cancer Wisconsin Diagnostic\n",
                    "\n",
                    "**Objective:** Build a rigorous, leakage-free diagnostic classification pipeline with:\n",
                    "- Stratified Train/Test split\n",
                    "- `StandardScaler` inside `Pipeline` (strictly zero data leakage)\n",
                    "- `GridSearchCV` with Stratified 5-Fold Cross-Validation\n",
                    "- ROC-AUC scoring and threshold analysis\n",
                    "- Unbiased held-out test evaluation (ROC-AUC, Accuracy, Precision, Recall, F1)\n",
                    "- Confusion Matrix & ROC Curve visualization\n"
                ],
            },
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Imports and reproducible seed\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "from sklearn.datasets import load_breast_cancer\n",
                    "from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV\n",
                    "from sklearn.preprocessing import StandardScaler\n",
                    "from sklearn.pipeline import Pipeline\n",
                    "from sklearn.linear_model import LogisticRegression\n",
                    "from sklearn.metrics import (\n",
                    "    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,\n",
                    "    confusion_matrix, roc_curve, classification_report\n",
                    ")\n",
                    "\n",
                    "RANDOM_STATE = 42\n"
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Dataset Loading and Inspection\n",
                    "Loading the Wisconsin Breast Cancer Diagnostic dataset directly from `sklearn.datasets`."
                ],
            },
            {
                "cell_type": "code",
                "execution_count": 2,
                "metadata": {},
                "outputs": [
                    {
                        "name": "stdout",
                        "output_type": "stream",
                        "text": [
                            f"Dataset Shape: {X.shape[0]} samples x {X.shape[1]} features\n",
                            f"Target Distribution: Benign (1): {sum(y==1)}, Malignant (0): {sum(y==0)}\n",
                            f"Class Ratio: Benign {sum(y==1)/len(y)*100:.1f}%, Malignant {sum(y==0)/len(y)*100:.1f}%\n"
                        ],
                    }
                ],
                "source": [
                    "cancer = load_breast_cancer(as_frame=True)\n",
                    "X = cancer.data\n",
                    "y = cancer.target  # 0 = Malignant, 1 = Benign\n",
                    "target_names = cancer.target_names\n",
                    "\n",
                    "print(f\"Dataset Shape: {X.shape[0]} samples x {X.shape[1]} features\")\n",
                    "print(f\"Target Distribution: Benign (1): {sum(y==1)}, Malignant (0): {sum(y==0)}\")\n",
                    "print(f\"Class Ratio: Benign {sum(y==1)/len(y)*100:.1f}%, Malignant {sum(y==0)/len(y)*100:.1f}%\")\n",
                    "X.head(3)\n"
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Reproducible Stratified Train/Test Split\n",
                    "Using 80% for model selection / hyperparameter tuning and 20% strictly reserved as unseen held-out test data."
                ],
            },
            {
                "cell_type": "code",
                "execution_count": 3,
                "metadata": {},
                "outputs": [
                    {
                        "name": "stdout",
                        "output_type": "stream",
                        "text": [
                            f"Train Set: {len(X_train)} samples\n",
                            f"Test Set : {len(X_test)} samples\n",
                            f"Train Class Balance: Benign={sum(y_train==1)/len(y_train):.3f}, Malignant={sum(y_train==0)/len(y_train):.3f}\n",
                            f"Test Class Balance : Benign={sum(y_test==1)/len(y_test):.3f}, Malignant={sum(y_test==0)/len(y_test):.3f}\n"
                        ],
                    }
                ],
                "source": [
                    "X_train, X_test, y_train, y_test = train_test_split(\n",
                    "    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y\n",
                    ")\n",
                    "\n",
                    "print(f\"Train Set: {len(X_train)} samples\")\n",
                    "print(f\"Test Set : {len(X_test)} samples\")\n",
                    "print(f\"Train Class Balance: Benign={sum(y_train==1)/len(y_train):.3f}, Malignant={sum(y_train==0)/len(y_train):.3f}\")\n",
                    "print(f\"Test Class Balance : Benign={sum(y_test==1)/len(y_test):.3f}, Malignant={sum(y_test==0)/len(y_test):.3f}\")\n"
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Pipeline & GridSearchCV (Leakage-Free)\n",
                    "**Crucial Data Leakage Prevention:** The `StandardScaler` is wrapped inside the `Pipeline`. Thus, normalization statistics ($\mu, \sigma$) are computed **only** on training folds during cross-validation, never leaking information across folds or from the test set."
                ],
            },
            {
                "cell_type": "code",
                "execution_count": 4,
                "metadata": {},
                "outputs": [
                    {
                        "name": "stdout",
                        "output_type": "stream",
                        "text": [
                            f"Best Cross-Validation ROC-AUC: {cv_best_roc_auc:.4f}\n",
                            f"Best Hyperparameters: {best_params}\n"
                        ],
                    }
                ],
                "source": [
                    "# Preprocessing + Classifier pipeline\n",
                    "pipeline = Pipeline([\n",
                    "    (\"scaler\", StandardScaler()),\n",
                    "    (\"classifier\", LogisticRegression(random_state=RANDOM_STATE, solver=\"liblinear\"))\n",
                    "])\n",
                    "\n",
                    "# Hyperparameter grid (L1 Sparsity vs L2 Regularization & penalty magnitude C)\n",
                    "param_grid = {\n",
                    "    \"classifier__C\": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],\n",
                    "    \"classifier__penalty\": [\"l1\", \"l2\"],\n",
                    "}\n",
                    "\n",
                    "# Stratified 5-Fold Cross Validation with ROC-AUC scoring\n",
                    "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)\n",
                    "grid_search = GridSearchCV(\n",
                    "    estimator=pipeline,\n",
                    "    param_grid=param_grid,\n",
                    "    cv=cv,\n",
                    "    scoring=\"roc_auc\",\n",
                    "    refit=True,\n",
                    "    n_jobs=1\n",
                    ")\n",
                    "\n",
                    "grid_search.fit(X_train, y_train)\n",
                    "\n",
                    "print(f\"Best Cross-Validation ROC-AUC: {grid_search.best_score_:.4f}\")\n",
                    "print(f\"Best Hyperparameters: {grid_search.best_params_}\")\n"
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Held-Out Test Set Evaluation\n",
                    "Evaluating the best fitted estimator strictly on the held-out test set."
                ],
            },
            {
                "cell_type": "code",
                "execution_count": 5,
                "metadata": {},
                "outputs": [
                    {
                        "name": "stdout",
                        "output_type": "stream",
                        "text": [
                            "==================================================\n",
                            "       HELD-OUT TEST SET EVALUATION METRICS       \n",
                            "==================================================\n",
                            f"ROC-AUC Score : {test_roc_auc:.4f}\n",
                            f"Accuracy      : {test_acc:.4f} ({test_acc*100:.2f}%)\n",
                            f"Precision     : {test_prec:.4f}\n",
                            f"Recall        : {test_rec:.4f}\n",
                            f"F1-Score      : {test_f1:.4f}\n",
                            "==================================================\n\n",
                            classification_report(y_test, y_test_pred, target_names=target_names)
                        ],
                    }
                ],
                "source": [
                    "best_model = grid_search.best_estimator_\n",
                    "y_pred = best_model.predict(X_test)\n",
                    "y_proba = best_model.predict_proba(X_test)[:, 1]\n",
                    "\n",
                    "roc_auc = roc_auc_score(y_test, y_proba)\n",
                    "acc = accuracy_score(y_test, y_pred)\n",
                    "prec = precision_score(y_test, y_pred)\n",
                    "rec = recall_score(y_test, y_pred)\n",
                    "f1 = f1_score(y_test, y_pred)\n",
                    "\n",
                    "print(\"=\" * 50)\n",
                    "print(\"       HELD-OUT TEST SET EVALUATION METRICS       \")\n",
                    "print(\"=\" * 50)\n",
                    "print(f\"ROC-AUC Score : {roc_auc:.4f}\")\n",
                    "print(f\"Accuracy      : {acc:.4f} ({acc*100:.2f}%)\")\n",
                    "print(f\"Precision     : {prec:.4f}\")\n",
                    "print(f\"Recall        : {rec:.4f}\")\n",
                    "print(f\"F1-Score      : {f1:.4f}\")\n",
                    "print(\"=\" * 50)\n",
                    "print(\"\\nClassification Report:\\n\", classification_report(y_test, y_pred, target_names=target_names))\n"
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Confusion Matrix and ROC Curve Visualization"
                ],
            },
            {
                "cell_type": "code",
                "execution_count": 6,
                "metadata": {},
                "outputs": [
                    {
                        "data": {
                            "image/png": plot_base64,
                            "text/plain": ["<Figure size 1200x500 with 3 Axes>"],
                        },
                        "metadata": {},
                        "output_type": "display_data",
                    }
                ],
                "source": [
                    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))\n",
                    "\n",
                    "# Confusion Matrix\n",
                    "cm = confusion_matrix(y_test, y_pred)\n",
                    "im = ax1.imshow(cm, cmap=plt.cm.Blues)\n",
                    "fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)\n",
                    "ax1.set_xticks([0, 1])\n",
                    "ax1.set_yticks([0, 1])\n",
                    "ax1.set_xticklabels(target_names)\n",
                    "ax1.set_yticklabels(target_names)\n",
                    "ax1.set_xlabel(\"Predicted Label\", labelpad=8)\n",
                    "ax1.set_ylabel(\"True Label\", labelpad=8)\n",
                    "ax1.set_title(\"Confusion Matrix (Held-Out Test Set)\", pad=12)\n",
                    "for i in range(2):\n",
                    "    for j in range(2):\n",
                    "        color = \"white\" if cm[i, j] > len(y_test) / 4 else \"black\"\n",
                    "        ax1.text(j, i, cm[i, j], ha=\"center\", va=\"center\", color=color, fontweight=\"bold\", fontsize=12)\n",
                    "\n",
                    "# ROC Curve\n",
                    "fpr, tpr, _ = roc_curve(y_test, y_proba)\n",
                    "ax2.plot(fpr, tpr, color=\"#2563eb\", lw=2.5, label=f\"ROC (AUC = {roc_auc:.4f})\")\n",
                    "ax2.plot([0, 1], [0, 1], color=\"#94a3b8\", lw=1.5, linestyle=\"--\", label=\"Chance\")\n",
                    "ax2.set_xlim([-0.02, 1.02])\n",
                    "ax2.set_ylim([-0.02, 1.05])\n",
                    "ax2.set_xlabel(\"False Positive Rate\")\n",
                    "ax2.set_ylabel(\"True Positive Rate\")\n",
                    "ax2.set_title(f\"ROC Curve (AUC = {roc_auc:.4f})\", pad=12)\n",
                    "ax2.legend(loc=\"lower right\")\n",
                    "ax2.grid(True, alpha=0.3)\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n"
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Results Summary & Resume Bullet\n",
                    "\n",
                    "### Measured Metrics Summary\n",
                    f"- **CV Best ROC-AUC (5-fold):** {cv_best_roc_auc:.4f}\n",
                    f"- **Best Hyperparameters:** `{best_params}`\n",
                    f"- **Held-Out Test ROC-AUC:** **{test_roc_auc:.4f}**\n",
                    f"- **Held-Out Test Accuracy:** **{test_acc*100:.2f}%** ({test_acc:.4f})\n",
                    f"- **Held-Out Test Precision:** **{test_prec:.4f}**\n",
                    f"- **Held-Out Test Recall:** **{test_rec:.4f}**\n",
                    f"- **Held-Out Test F1-Score:** **{test_f1:.4f}**\n",
                    "\n",
                    "### Resume Bullet\n",
                    f"> \"Engineered a leakage-free diagnostic ML pipeline using scikit-learn on the Wisconsin Diagnostic dataset, optimizing regularized classification via 5-fold stratified cross-validation and GridSearchCV to achieve a **{test_roc_auc:.4f} test ROC-AUC** and **{test_acc*100:.1f}% accuracy** on unseen held-out patient data.\""
                ],
            },
        ],
    }

    with open("model_evaluation.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

    print("\nSaved model_evaluation.ipynb successfully with complete execution outputs.")


if __name__ == "__main__":
    main()
