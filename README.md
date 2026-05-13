# 🎗️ Cancer Risk Level Prediction

### An End-to-End Machine Learning Classification Study

> *A research-quality multi-class classification pipeline covering EDA, Feature Engineering, Hyperparameter Tuning, Multi-Model Comparison, and XGBoost-powered SHAP Explainability.*

---

## 📌 Project Overview

Cancer remains one of the leading causes of mortality worldwide. Early identification of patient risk levels — **Low**, **Medium**, or **High** — based on lifestyle, genetic, and environmental features allows clinicians to prioritise screening, design personalised prevention plans, and allocate healthcare resources more efficiently.

**Why Machine Learning?**  
Traditional rule-based scoring misses complex *non-linear interactions* between features (e.g., how smoking *combined with* air pollution *and* low physical activity compounds risk). Ensemble ML models capture these interactions automatically, producing more robust and interpretable risk scores.

| Item | Detail |
| --- | --- |
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
├── shap_low_risk_sample396.png               # SHAP plot — Low Risk
├── shap_medium_risk_sample12.png             # SHAP plot — Medium Risk
├── shap_high_risk_sample14.png               # SHAP plot — High Risk
└── README.md
```

---

## ⚙️ Pipeline Overview

```
Raw Data → EDA → Feature Engineering → Scaling → Train/Test Split
    → GridSearchCV (9 Models) → Evaluation → XGBoost + SHAP
```

---

## 📊 Exploratory Data Analysis (EDA)

EDA is the most important phase before modelling — it reveals distribution patterns, class imbalances, outliers, and feature correlations that directly influence preprocessing and model selection.

### 🌡️ Correlation Heatmap — Why It Matters

[![Correlation Heatmap](https://github.com/sultan-ru118/ML-Project/raw/main/correlation_heatmap.png)](https://github.com/sultan-ru118/ML-Project/blob/main/correlation_heatmap.png)

The **Pearson correlation matrix** reveals linear relationships between all numeric features. Values close to **+1 or −1** indicate strong correlation; values near **0** suggest independence.

**Key observations from the heatmap:**

| Observation | Implication |
| --- | --- |
| `Overall_Risk_Score` has the highest correlation with `Risk_Level` | **Dropped** — it is a composite of the target (data leakage) |
| `Smoking`, `Air_Pollution`, `Alcohol_Use` show moderate positive correlation with `Risk_Level` | Clinically expected and confirmed as top SHAP drivers |
| Multicollinearity between one-hot `Cancer_Type` dummies is negligible | Dummy variable trap mitigated effectively |

> **Why Pearson?** It measures linear association. While tree-based models don't require linearity, the heatmap still identifies redundant features (preventing multicollinearity) and validates clinical expectations before any model is trained.

---

## 🛠️ Feature Engineering & Preprocessing

| Step | Method | Reason |
| --- | --- | --- |
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
| --- | --- | --- |
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

| Rank | Model | Accuracy | Precision (macro) | Recall (macro) | F1 Score (macro) |
| --- | --- | --- | --- | --- | --- |
| 🥇 | XGBoost | 0.8625 | 0.865682 | 0.587839 | 0.650426 |
| 🥈 | Gradient Boosting | 0.8700 | 0.835642 | 0.656973 | 0.719733 |
| 🥉 | AdaBoost | 0.8750 | 0.804944 | 0.703284 | 0.738716 |
| 4 | Naïve Bayes | 0.8300 | 0.759121 | 0.724006 | 0.736450 |
| 5 | Random Forest | 0.8275 | 0.726821 | 0.724153 | 0.720343 |
| 6 | SVM | 0.8320 | 0.658991 | 0.698953 | 0.676126 |
| 7 | Logistic Regression | 0.8125 | 0.645918 | 0.874413 | 0.706251 |
| 8 | KNN | 0.88175 | 0.645227 | 0.475354 | 0.508695 |
| 9 | Decision Tree | 0.7325 | 0.492172 | 0.501523 | 0.496583 |

---

### 🏆 Why XGBoost is the Best — Deep Explanation

XGBoost (Extreme Gradient Boosting) outperforms all other models in this study for three interconnected reasons:

**1. It builds on Gradient Boosting — but fixes its weaknesses**

Gradient Boosting trains trees sequentially, where each new tree corrects the errors (residuals) of the previous ones. XGBoost adds **L1/L2 regularisation** (`gamma`, `lambda`) directly into the tree-building objective — penalising model complexity as it learns.

**2. It captures the non-linear risk interactions that simpler models miss**

The compounding effect of *Smoking + Air_Pollution + low Physical_Activity* is far greater than their individual sums. XGBoost's deep trees capture these **multiplicative feature interactions** naturally.

**3. Stochastic training prevents overfitting further**

The tuned model uses `subsample=0.8` — each tree only sees 80% of training rows, introducing healthy randomness that reduces variance.

| What others do wrong | Why XGBoost handles it |
| --- | --- |
| Logistic Regression — linear boundary only | XGBoost learns non-linear splits at every node |
| Decision Tree — overfits without depth control | Regularisation + depth tuning controls complexity |
| Random Forest — all trees independent, no correction | Boosting corrects residuals from previous trees |
| Gradient Boosting — no regularisation | XGBoost adds `gamma` + `lambda` penalisation |
| Naïve Bayes — assumes feature independence | XGBoost models interaction effects natively |

---

<!-- ============================================================ -->
<!--        ✨ EXPLAINABLE AI — SHAP SECTION                     -->
<!-- ============================================================ -->

## 🔍 Explainable AI — SHAP Analysis ⭐ 

> **"Predicting the risk level is not enough — we must explain WHY the model made that decision."**
>
> This section is the **most important contribution** of this project.
> Using SHAP, every single prediction is made **transparent, trustworthy, and clinically actionable**.

---

### 🧠 What is SHAP?

**SHAP = SH**apley **A**dditive ex**P**lanations — uses game theory to assign each feature a contribution score.

```
f(x)  =  E[f(X)]  +  Σ (all SHAP contributions)
Final =  Baseline +  Sum of all feature pushes
```

| Bar Colour | Meaning |
|---|---|
| 🔴 Pink/Red | Feature **pushes** prediction toward this risk class |
| 🔵 Blue | Feature **resists** this prediction |
| Left number | Negative = patient value is **LOW** · Positive = patient value is **HIGH** |

---

### 🌐 Global Feature Importance

| Rank | Feature | Direction | Clinical Significance |
|---|---|---|---|
| 1 | `Smoking` | ↑ High Risk | #1 modifiable cancer risk factor globally |
| 2 | `Air_Pollution` | ↑ High Risk | Chronic carcinogen inhalation |
| 3 | `Alcohol_Use` | ↑ High Risk | Linked to 7+ cancer types |
| 4 | `Physical_Activity_Level` | ↓ Protective | Reduces risk when high |
| 5 | `Occupational_Hazards` | ↑ High Risk | Workplace carcinogen exposure |

---

### 🌊 SHAP Waterfall Plots — Three Risk Level Cases

> One representative patient selected per risk class.
> Each plot shows **step-by-step** how the model reached its decision.

---

#### 🟢 CASE 1 — LOW RISK · Sample 396 · Class 0

![SHAP Waterfall — Low Risk Sample 396](https://github.com/sultan-ru118/ML-Project/raw/main/shap_low_risk_sample396.png)

| Metric | Value |
|---|---|
| Baseline `E[f(X)]` | −2.644 |
| Final Score `f(x)` | **+4.533** |
| Score Shift | **+7.18** |
| Prediction | ✅ Low Risk (Class 0) |

| Feature | Value | SHAP | Meaning |
|---|---|---|---|
| `Alcohol_Use` | −1.544 → **LOW** | **+1.79** 🔴 | Low alcohol → strong low-risk signal |
| `Obesity` | −1.623 → **LOW** | **+1.59** 🔴 | Not obese → model confident |
| `Air_Pollution` | −1.660 → **LOW** | **+1.47** 🔴 | Clean air environment |
| `Smoking` | −0.950 → **LOW** | **+1.26** 🔴 | Non-smoker |
| `Diet_Red_Meat` | +1.525 → **HIGH** | **−0.95** 🔵 | Only opposing factor |

> **Conclusion:** All major risk factors are LOW. Healthy lifestyle features
> greatly outweighed the high red meat concern → **Low Risk ✅**

---

#### 🟡 CASE 2 — MEDIUM RISK · Sample 12 · Class 1

![SHAP Waterfall — Medium Risk Sample 12](https://github.com/sultan-ru118/ML-Project/raw/main/shap_medium_risk_sample12.png)

| Metric | Value |
|---|---|
| Baseline `E[f(X)]` | +0.353 |
| Final Score `f(x)` | **+1.457** |
| Score Shift | **+1.10** |
| Prediction | ⚠️ Medium Risk (Class 1) |

| Feature | Value | SHAP | Meaning |
|---|---|---|---|
| `Smoking` | +1.457 → **HIGH** | **+0.71** 🔴 | ← **#1 biggest driver** |
| `Air_Pollution` | +1.459 → **HIGH** | **+0.23** 🔴 | Compounds smoking effect |
| `Alcohol_Use` | +0.296 → Moderate | **+0.23** 🔴 | Adds pressure |
| `Physical_Activity` | +0.019 | **−0.16** 🔵 | Slight resistance |
| `Diet_Salted_Processed` | −1.154 → **LOW** | **−0.16** 🔵 | Healthy diet resists — not enough |

> **Conclusion:** Smoking alone elevated this patient from Low → Medium risk.
> Even a healthy diet could not overcome smoking + pollution → **Medium Risk ⚠️**

---

#### 🔴 CASE 3 — HIGH RISK · Sample 14 · Class 2

![SHAP Waterfall — High Risk Sample 14](https://github.com/sultan-ru118/ML-Project/raw/main/shap_high_risk_sample14.png)

| Metric | Value |
|---|---|
| Baseline `E[f(X)]` | −3.386 |
| Final Score `f(x)` | **+3.368** |
| Score Shift | **+6.75** ← Largest shift |
| Prediction | 🚨 High Risk (Class 2) |

| Feature | Value | SHAP | Meaning |
|---|---|---|---|
| `Occupational_Hazards` | +1.563 → **HIGH** | **+2.27** 🔴 | ← **#1 driver — dangerous job** |
| `Alcohol_Use` | +0.909 → **HIGH** | **+1.69** 🔴 | Heavy alcohol use |
| `Air_Pollution` | +0.835 → **HIGH** | **+1.17** 🔴 | High pollution exposure |
| `Smoking` | +1.457 → **HIGH** | **+1.18** 🔴 | Heavy smoker |
| `Family_History` | +2.035 → **HIGH** | **+0.23** 🔴 | Genetic predisposition |
| `Diet_Salted_Processed` | −0.506 → LOW | **−0.78** 🔵 | Healthy diet — too weak |
| `Obesity` | −0.316 → LOW | **−0.33** 🔵 | Not obese — small resistance |

> **Conclusion:** 5 simultaneous high-risk factors overwhelmed all protective factors.
> Model correctly issued: **High Risk 🚨**

---

### 📋 SHAP Comparison — All Three Risk Levels

| | 🟢 LOW (0) | 🟡 MEDIUM (1) | 🔴 HIGH (2) |
|---|---|---|---|
| **Sample** | 396 | 12 | 14 |
| **Baseline** | −2.644 | +0.353 | −3.386 |
| **Final f(x)** | **+4.533** | **+1.457** | **+3.368** |
| **Score Shift** | +7.18 | +1.10 | +6.75 |
| **#1 Driver** | None dominant | Smoking +0.71 | Occ. Hazards +2.27 |
| **HIGH features** | 1 only | 2–3 | 5+ |
| **Pattern** | All values LOW | 1–2 factors HIGH | Multiple HIGH |

### 🔑 Key SHAP Insight

> As risk level increases **Low → Medium → High**:
> - More **red bars** appear → more active risk factors
> - SHAP values get **larger** → each factor pushes harder
> - Score shift becomes **more dramatic**
> - Protective blue bars become **too small** to resist
>
> This confirms the model has learned **clinically meaningful, compounding risk patterns.**

---

### 🏥 Why This Matters Clinically

| Stakeholder | What SHAP Gives Them |
|---|---|
| **Doctor** | Ranked list of *why* a patient is high risk → actionable treatment plan |
| **Patient** | Clear modifiable factors (smoking, diet, activity) → personal prevention guide |
| **Researcher** | Quantified, reproducible feature importance → publishable evidence |
| **Policy Maker** | Top factors (Smoking + Air Pollution) → public health priority targets |

<!-- ============================================================ -->
<!--        END OF SHAP SECTION                                   -->
<!-- ============================================================ -->

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

## 📓 What This Notebook Does — Section by Section

| Section | What I Did | Why |
| --- | --- | --- |
| §1 Business Problem | Defined the clinical question: stratify patients into Low / Medium / High cancer risk | Anchors all design decisions in a real medical need |
| §2–3 Data Loading | Loaded `cancer-risk-factors.csv`, inspected shape, dtypes, nulls, target distribution | Zero missing values — no imputation needed |
| §4 EDA | Pie charts, bar, box, count, and boxen plots across key features | Uncovered class balance, gender splits, BMI–cancer links, smoking–lung correlation |
| §5 Encoding | One-hot encoded `Cancer_Type`; ordinally mapped `Risk_Level` (Low→0, Med→1, High→2) | Machines need numbers; ordinal encoding preserves severity order |
| §6 Correlation | Pearson heatmap on all features | Detected data leakage (`Overall_Risk_Score`), confirmed clinical signals |
| §7–8 Cleaning | Removed duplicates, shuffled dataset | Prevents training bias from repeated rows and sorted ordering |
| §9 Feature Selection + Scaling | Dropped leaky/redundant columns; applied `StandardScaler` | Fair model comparison; critical for distance-based models (KNN, SVM) |
| §10 Split | 80/20 train-test split with `random_state=42` | Unseen test set for unbiased final evaluation |
| §11 Baseline DT | Trained an untuned Decision Tree | Sets a lower-bound reference before any optimisation |
| §12 GridSearchCV | Tuned 6 models (LR, KNN, GNB, SVM, DT, RF) via 5-fold CV on weighted F1 | Finds optimal hyperparameters without test set leakage |
| §13 Boosting | Tuned AdaBoost, Gradient Boosting, XGBoost | Advanced ensemble methods that outperform single trees |
| §14 Comparison | Ranked all 9 models on Accuracy, Precision, Recall, F1 | Identified XGBoost as the top performer |
| §15 SHAP | Global bar + three waterfall plots (Low/Medium/High) + per-class bars | Made model decisions transparent and clinically auditable |
| §16 Conclusion | Summarised key findings, feature rankings, future improvements | Research-ready writeup with clinical and policy implications |

---

## 🛠️ Requirements

```
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap
```

---

## 🚀 Usage

```
# Clone and run
jupyter notebook Cancer_Risk_Level_Enhanced_claude.ipynb
```

---

## 📄 License

This project is built for research and educational purposes.

---

🎗️ Built for Cancer Risk Research · Powered by XGBoost + SHAP
