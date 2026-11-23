# 📉 Customer Churn Prediction — Gradient Boosting & Ensemble Blending

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/scikit--learn-Pipelines-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/CatBoost-Boosting-FFCC00" alt="CatBoost">
  <img src="https://img.shields.io/badge/XGBoost-Boosting-189AB4" alt="XGBoost">
  <img src="https://img.shields.io/badge/LightGBM-Boosting-2E8B57" alt="LightGBM">
  <img src="https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle&logoColor=white" alt="Kaggle">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

<p align="center">
  A Kaggle competition solution that predicts <b>customer churn</b> probabilities using a clean
  preprocessing pipeline, tuned gradient boosting models, and weighted ensemble blending — optimized
  for <b>ROC AUC</b>.
</p>

---

## 📖 Overview

This project predicts whether a customer will churn using demographic, account, and service usage data.
The goal is a robust machine learning pipeline that outputs **binary churn probabilities** and maximizes
the **Area Under the ROC Curve (ROC AUC)**.

The notebook covers:

- 🧹 **Cleaning** — handling missing values and preparing the target
- 🔍 **Exploration** — chi-square analysis of categorical features against churn
- ⚙️ **Pipeline** — `ColumnTransformer` with scaling and one-hot encoding
- 🤖 **Modeling** — CatBoost, XGBoost, and LightGBM with 5-fold cross-validation
- 🎛️ **Tuning** — `RandomizedSearchCV` hyperparameter optimization
- 🧩 **Ensembling** — weighted blends for the final Kaggle submissions

---

## 🗂️ Dataset

| Detail | Description |
|---|---|
| **Source** | <Kaggle competition name / link> |
| **Train** | 7,849 rows × 21 columns |
| **Test** | 7,900 rows × 20 columns (no `Churn` column) |
| **Target** | `Churn` (converted to binary `Churn_binary`: 0/1) |
| **Key columns** | `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `InternetService`, `PaymentMethod`, `SeniorCitizen` |

> 📥 The data is not included in this repo. Download it from the competition page and place it in the
> working directory before running the notebook.

---

## 🧹 Data Cleaning & Exploration

1. **Dropped a single incomplete record** with missing values across multiple columns in the training set
2. **Converted the target** `Churn` to binary values `(0, 1)` stored as `Churn_binary`
3. **Ran chi-square analysis** on categorical features — `InternetService`, `PaymentMethod`, and
   `Contract` showed the strongest association with churn

---

## ⚙️ Preprocessing Pipeline

A structured Scikit-Learn `ColumnTransformer` handles both feature types:

| Feature type | Columns | Transformation |
|---|---|---|
| **Numerical** | `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges` | `StandardScaler` |
| **Categorical** | Demographics and service indicators | `OneHotEncoder(handle_unknown='ignore')` |

---

## 🤖 Model Evaluation & Tuning

Three gradient boosting frameworks were compared with **5-fold cross-validation**.

### Initial CV Performance (Accuracy)

| Model | Accuracy | Std |
|---|---|---|
| **CatBoost** | 0.8421 | 0.0027 |
| **XGBoost** | 0.8367 | 0.0052 |
| **LightGBM** | 0.8290 | 0.0038 |

### Hyperparameter Tuning (`RandomizedSearchCV`)

| Model | Best parameters | Best CV ROC AUC |
|---|---|---|
| **CatBoost** | `depth=4`, `iterations=300`, `l2_leaf_reg=1`, `learning_rate=0.05` | <add> |
| **LightGBM** | `num_leaves=20`, `n_estimators=300`, `max_depth=6`, `learning_rate=0.01`, `reg_alpha=0.3`, `reg_lambda=0.1` | **0.9081** |
| **XGBoost** | `subsample=0.7`, `n_estimators=1000`, `max_depth=3`, `learning_rate=0.01`, `lambda=0.5`, `gamma=0.3` | **0.9107** |

---

## 🧩 Ensemble Blending & Submissions

| Strategy | Weighting | Training ROC AUC | Output |
|---|---|---|---|
| **A — 2-model blend** | 0.6 × CatBoost + 0.4 × LightGBM | 0.9284 | `submission_tuned.csv` |
| **B — 3-model ensemble** | 0.3 × CatBoost + 0.3 × LightGBM + 0.4 × XGBoost | 0.9270 | `submission_ensemble.csv` |

**Leaderboard result:** <public/private score and rank>

---

## 🛠️ Tech Stack

- **Python** — pandas, numpy
- **Modeling** — scikit-learn, CatBoost, XGBoost, LightGBM
- **Statistics** — scipy (chi-square tests)
- **Visualization** — matplotlib, seaborn
- **Environment** — developed in Google Colab / Jupyter / Kaggle Notebooks

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/jaminjuma/<repo-name>.git
cd <repo-name>

# Install dependencies
pip install pandas numpy scipy matplotlib seaborn scikit-learn catboost xgboost lightgbm

# Download the competition data (train.csv, test.csv) into the project folder,
# then run the notebook
jupyter notebook
```

---

## 🔮 Possible Next Steps

- Optimize blend weights with out-of-fold predictions instead of hand-picked weights
- Add stacking with a logistic regression meta-model
- Feature engineering (e.g., tenure buckets, charges-per-month ratios, service counts)
- Model explainability with SHAP to show what drives churn
- Threshold tuning for business-specific precision/recall trade-offs

---

## 👤 Author

**Jamin Juma**
Founder of [Automateke](https://automateke.app) · CS student, Kirinyaga University

- 🌐 Portfolio: [jaminjuma.tech](https://jaminjuma.tech)
- 💻 GitHub: [@jaminjuma](https://github.com/jaminjuma)
- 🔗 LinkedIn: [jumajamin](https://linkedin.com/in/jumajamin)

---

## 📄 License

This project is licensed under the MIT License — feel free to use and adapt it.
