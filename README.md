# auto-ml-model-selector
This repository provides a Python tool that automates the benchmarking of multiple machine learning models. Instead of manually experimenting with individual algorithms, this script streamlines the process by preprocessing the dataset, training various models, and reporting performance in a clear leaderboard format.  
# Auto ML Model Benchmark 🚀

A Python tool to **automatically benchmark multiple ML models** (classification & regression) on any dataset, and recommend the best model based on performance.

## ✨ Features
- Detects **classification vs regression** automatically
- Preprocesses data:
  - Missing value imputation
  - Label encoding for categoricals
  - Train-only scaling (no leakage)
- Benchmarks 12+ algorithms:
  - Logistic Regression, SVM, KNN, Decision Tree, Random Forest, Naive Bayes, Gradient Boosting, Neural Network, and XGBoost (if installed)
- Evaluates with:
  - Classification → Accuracy, Precision, Recall, F1, ROC-AUC
  - Regression → MAE, MSE, RMSE, R²
- Outputs a **leaderboard** and highlights the 🏆 best model
- CLI support for running on any CSV

## 📦 Installation
Clone this repo:
```bash
git clone https://github.com/<your-username>/ml-model-benchmark.git
cd ml-model-benchmark
pip install -r requirements.txt
