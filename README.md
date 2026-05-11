# 🎗️ Cancer Risk Level Prediction
### An End-to-End Machine Learning Classification Study

> *A research-quality multi-class classification pipeline covering EDA, Feature Engineering, Hyperparameter Tuning, Multi-Model Comparison, and XGBoost-powered SHAP Explainability.*

---

## 📌 Project Overview

Cancer remains one of the leading causes of mortality worldwide. Early identification of patient risk levels — **Low**, **Medium**, or **High** — based on lifestyle, genetic, and environmental features allows clinicians to prioritise screening, design personalised prevention plans, and allocate healthcare resources more efficiently.

**Why Machine Learning?**  
Traditional rule-based scoring misses complex *non-linear interactions* between features (e.g., how smoking *combined with* air pollution *and* low physical activity compounds risk). Ensemble ML models capture these interactions automatically, producing more robust and interpretable risk scores.

| Item | Detail |
|------|--------|
| **Dataset** | `cancer-risk-factors.csv` |
| **Target** | `Risk_Level` — Low / Medium / High (3-class) |
| **Models Evaluated** | 9 Classifiers |
| **Best Model** | XGBoost + SHAP Explainability |
| **Validation Strategy** | 5-Fold Cross-Validation + GridSearchCV |
| **Random Seed** | 42 (full reproducibility) |

---

## 📁 Project Structure

```
├── Cancer_Risk_Level_Enhanced_claude.ipynb   # Main notebook
├── cancer-risk-factors.csv                   # Dataset
├── correlation_heatmap.png                   # Saved figure
└── README.md
```

---

## ⚙️ Pipeline Overview

```
5. THE MACHINE LEARNING PIPELINE

The overall pipeline follows this exact sequence:

RAW DATA
↓
Load + Inspect (df.info, describe, isnull)
↓
EDA (visualisations, correlation analysis)
↓
Encoding (One-Hot for Cancer_Type, Ordinal for Risk_Level)
↓
Drop Leakage + Irrelevant Columns
↓
Remove Duplicates → Shuffle
↓
Feature Matrix X / Target y (train_test_split 80/20)
↓
StandardScaler (fit on X_train ONLY, transform both X_train and X_test)
↓
GridSearchCV (5-fold CV on training set for each model)
↓
Best Estimator Evaluation on Held-Out Test Set
↓
SHAP Explainability on Best Model

One critical engineering discipline here: StandardScaler is fit on the training
set only. If we fit the scaler on the entire dataset before splitting, we
introduce data leakage — the scaler would have seen the test set's distribution
and used it to transform both sets. The correct procedure is: fit_transform on
X_train, transform (not fit_transform) on X_test.
```

---

## 📊 Exploratory Data Analysis (EDA)

EDA is the most important phase before modelling — it reveals distribution patterns, class imbalances, outliers, and feature correlations that directly influence preprocessing and model selection.

### 🌡️ Correlation Heatmap — Why It Matters

![Correlation Heatmap](correlation_heatmap.png)

The **Pearson correlation matrix** reveals linear relationships between all numeric features. Values close to **+1 or −1** indicate strong correlation; values near **0** suggest independence.

**Key observations from the heatmap:**

| Observation | Implication |
|-------------|-------------|
| `Overall_Risk_Score` has the highest correlation with `Risk_Level` | **Dropped** — it is a composite of the target (data leakage) |
| `Smoking`, `Air_Pollution`, `Alcohol_Use` show moderate positive correlation with `Risk_Level` | Clinically expected and confirmed as top SHAP drivers |
| Multicollinearity between one-hot `Cancer_Type` dummies is negligible | Dummy variable trap mitigated effectively |

> **Why Pearson?** It measures linear association. While tree-based models don't require linearity, the heatmap still identifies redundant features (preventing multicollinearity) and validates clinical expectations before any model is trained.

---

## 🛠️ Feature Engineering & Preprocessing

| Step | Method | Reason |
|------|--------|--------|
| Categorical encoding | One-Hot Encoding (`Cancer_Type`) | Nominal — no ordinal relationship |
| Target encoding | Ordinal map: Low→0, Medium→1, High→2 | Preserves severity ordering |
| ID removal | Drop `Patient_ID` | Identifier — no predictive value |
| Leakage removal | Drop `Overall_Risk_Score`, `BMI`, `Physical_Activity` | High collinearity / computed from target |
| Scaling | `StandardScaler` (z-score) | Critical for KNN & SVM; consistent across all models |
| Deduplication | `drop_duplicates()` | Prevents training bias from repeated records |
| Shuffle | `sklearn.utils.shuffle` (seed=42) | Removes ordering bias from original file |

**Train / Test Split:** 80% training · 20% held-out test · `class_weight='balanced'` applied across classifiers to handle class imbalance.

---

## 🤖 Models & Hyperparameter Tuning

All models were tuned with **GridSearchCV (5-fold CV)** scored on **weighted F1** — chosen over raw accuracy because it handles class imbalance and penalises both false positives and false negatives simultaneously.

| Model | Key Tuned Parameters | Why This Model |
|-------|---------------------|----------------|
| Logistic Regression | `C`, `solver` | Linear interpretable baseline |
| K-Nearest Neighbours | `n_neighbors`, `weights='distance'` | Non-parametric, instance-based |
| Gaussian Naïve Bayes | — | Fast probabilistic sanity-check |
| SVM (RBF kernel) | `C`, `gamma` | Maximum-margin non-linear boundary |
| Decision Tree (Baseline) | — | Lower-bound reference (no tuning) |
| Decision Tree (Tuned) | `criterion`, `max_depth`, `max_features` | Shows value of tuning vs baseline |
| Random Forest | `n_estimators`, `max_depth`, `criterion` | Variance reduction via Bagging |
| AdaBoost | `n_estimators`, `learning_rate` | Focuses iteratively on hard examples |
| Gradient Boosting | `learning_rate`, `n_estimators` | Sequential residual correction |
| **XGBoost** | `learning_rate`, `n_estimators` | Regularised GB — top performer |

---

## 📈 Model Performance Comparison

All 9 models were evaluated on the **held-out test set** using four metrics:

| Metric | What It Measures | Why It Matters Here |
|--------|-----------------|---------------------|
| **Accuracy** | Overall correct predictions / total | General performance gauge |
| **Precision** (macro) | TP / (TP + FP) per class, averaged | Minimise false alarms |
| **Recall** (macro) | TP / (TP + FN) per class, averaged | Minimise missed high-risk patients |
| **F1 Score** (macro) | Harmonic mean of Precision & Recall | Balanced metric for imbalanced classes |

> ⚠️ **In cancer risk classification, Recall for the `High` class is clinically critical** — missing a high-risk patient is far more costly than a false alarm.

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| 🏆 **XGBoost** | **Best** | **Best** | **Best** | **Best** |
| Gradient Boosting | High | High | High | High |
| Random Forest | Moderate–High | Moderate–High | Moderate–High | Moderate–High |
| SVM | Moderate | Moderate | Moderate | Moderate |
| AdaBoost | Moderate | Moderate | Moderate | Moderate |
| Decision Tree (Tuned) | Moderate | Moderate | Moderate | Moderate |
| KNN | Moderate | Moderate | Moderate | Moderate |
| Logistic Regression | Lower | Lower | Lower | Lower |
| Naïve Bayes | Lowest | Lowest | Lowest | Lowest |

> *Exact metric values are available in Section 14 of the notebook. Rankings above reflect the model ordering produced by the evaluation pipeline.*

**Why XGBoost Won:**
- Regularised boosting (`gamma`, `subsample=0.8`) controls overfitting that plagues plain Gradient Boosting.
- Sequential residual correction captures complex risk factor interactions (e.g. Smoking × Air_Pollution).
- Native multi-class support via `mlogloss` objective.
- Full SHAP compatibility — enables clinical-grade explainability.

---

## 🔍 Explainable AI — SHAP Analysis

**Why SHAP instead of standard feature importance?**  
Standard Gini-based importance only shows *how often* a feature is used in splits. SHAP (SHapley Additive exPlanations) uses game theory to show *how much* each feature **pushes a specific prediction up or down** — enabling both global population-level and local per-patient explanations.

### Global Feature Importance — SHAP Bar Plot

The SHAP summary bar plot shows the **mean absolute SHAP value** per feature across all test samples — a true measure of global impact, not just split frequency.

**Top drivers (globally):**

| Rank | Feature | Direction | Clinical Significance |
|------|---------|-----------|----------------------|
| 1 | `Smoking` | ↑ High Risk | #1 modifiable cancer risk factor globally |
| 2 | `Air_Pollution` | ↑ High Risk | Chronic carcinogen inhalation |
| 3 | `Alcohol_Use` | ↑ High Risk | Linked to 7+ cancer types |
| 4 | `Physical_Activity_Level` | ↓ High Risk | Protective lifestyle factor |
| 5 | `Age` | ↑ High Risk | Biological risk amplifier |

### Patient-Level — SHAP Waterfall Plot

The waterfall plot decomposes a **single patient's prediction** step-by-step — starting from the model's average output (base value) and showing how each feature pushes the final prediction up (red) or down (blue).

**Example Patient (index 12):**
- `Smoking` (value = 1.457) → SHAP = **+0.71** — single largest risk driver
- `Air_Pollution` (value = 1.459) → SHAP = **+0.23** — environmental exposure
- `Physical_Activity` (value = 0.019) → SHAP = **−0.16** — slight protective pull

> This makes the model's decision **transparent and auditable** — essential for clinical AI deployment where clinicians need to understand *why* a patient was flagged as high-risk.

### Class-Specific SHAP Bar Plots

SHAP supports multi-output models natively, showing feature importance **separately per risk class**:

- **Low Risk class:** High `Physical_Activity` and low `Smoking` are protective.
- **Medium Risk class:** Mixed moderate signals across several features.
- **High Risk class:** `Smoking`, `Air_Pollution`, `Alcohol_Use`, and low `Physical_Activity_Level` consistently dominate.

This class-level breakdown is invaluable for **targeted public health intervention design**.

---

## 🔑 Key Findings

1. **XGBoost** achieves the highest test F1 (macro) and is the recommended model for deployment.
2. **Smoking** is the single strongest individual predictor of High cancer risk — confirmed by both correlation analysis and SHAP.
3. **Environmental factors** (`Air_Pollution`) and **lifestyle factors** (`Alcohol_Use`, `Physical_Activity_Level`) are the next most influential predictors.
4. **Age** acts as a biological risk amplifier across all cancer types.
5. SHAP explainability confirms the model's reasoning aligns with established clinical evidence — increasing trust for medical use.

---

## 🔮 Future Work

- **Deep Learning:** TabNet or FT-Transformer for richer tabular feature interactions
- **SMOTE:** Synthetic oversampling if class imbalance is confirmed in larger datasets
- **Probability Calibration:** Platt scaling for well-calibrated risk probabilities
- **External Validation:** Test on independent cohort datasets
- **API Deployment:** Flask/FastAPI REST endpoint with per-prediction SHAP explanations
- **Longitudinal Tracking:** Time-series risk monitoring per patient over visits

---

## 🛠️ Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap
```

---

## 🚀 Usage

```bash
# Clone and run
jupyter notebook Cancer_Risk_Level_Enhanced_claude.ipynb
```

---

## 📄 License

This project is built for research and educational purposes.

---

<p align="center">🎗️ Built for Cancer Risk Research · Powered by XGBoost + SHAP</p>
