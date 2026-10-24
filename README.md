# smartphone-addiction-prediction
Advanced ML pipeline for Kaggle Playground Series S6E8 (Smartphone Addiction Prediction) achieving 0.9679 OOF AUC via feature engineering, multi-model stacking, and pseudo-labeling.

# Beating a 0.964 Baseline: Smartphone Addiction Prediction
> Kaggle Playground Series S6E8 | Advanced Gradient Boosting, Stacking, and Pseudo-Labeling

A comprehensive machine learning pipeline built for the Kaggle Playground Series S6E8 competition. The objective was to predict whether a person is addicted to their smartphone (`addicted_label`) based on behavioral, usage, and demographic data.

- **Dataset Scale:** 691,369 training rows and 296,302 test rows.
- **Evaluation Metric:** ROC-AUC.
- **Final Performance:** Reached an out-of-fold (OOF) AUC of **0.96799** using multi-seed bagged LightGBM and stacking.

---

##  Key Methodology & Pipeline Steps

### 1. Exploratory Data Analysis & Missingness as Signal
- Inspected shapes, data types, duplicates, and target balance.
- Noted that every feature had missing values ranging between 4% and 19%.
- Instead of blind imputation, missingness itself was treated as a signal by adding per-column missing flags and a total-missing count, while letting tree models handle NaNs natively.

### 2. Behavioral Feature Engineering
- Created ratio and interaction features to capture underlying usage patterns beyond raw hour counts:
  - Social media and gaming share of total screen time.
  - Screen-to-sleep and leisure-vs-work ratios.
  - Weekend vs. weekday screen time drift.
  - Notifications per app open and sleep deficit.
  - **Residual Feature (`other_screen_time`):** Captured unexplained screen time by subtracting social, gaming, and work hours from total daily screen time.

### 3. Leakage-Safe Validation & Baseline
- Trained LightGBM, XGBoost, and CatBoost models on GPU using **5-fold Stratified K-Fold** with early stopping.
- Relied strictly on Out-Of-Fold (OOF) predictions to ensure a leakage-free estimate of real-world generalization, achieving a baseline OOF AUC of **~0.9639**.

### 4. What Moved the Score (Advanced Improvements)
- **Value-Level Target Encoding:** Applied smoothed target statistics on exact rounded numeric values strictly inside each fold to prevent data leakage.
- **Higher Model Capacity:** Increased LightGBM/XGBoost capacity (e.g., 255 leaves, learning rate 0.02, up to 4000 rounds with early stopping), yielding a major performance jump.
- **Pseudo-Labeling:** Leveraged high-confidence test predictions (predictions >= 0.97 or <= 0.03 covering ~47.9% of the test set) as auxiliary training data, while keeping OOF evaluation strictly bound to real training rows.
- **Multi-Seed Bagging:** Trained LightGBM across multiple random seeds and averaged predictions to reduce variance, achieving a final OOF AUC of **0.96799**.

---

##  Results Summary

| Model / Approach | OOF AUC Score |
| :--- | :--- |
| Baseline (35 Features) | 0.96389 |
| LightGBM (v2 Capacity + Residual + Value TE) | 0.96766 |
| XGBoost (v2 Capacity) | 0.96773 |
| CatBoost | 0.96687 |
| Final Stack (`LGB + XGB + CAT`) | 0.96777 |
| **Final Bagged LightGBM (V3 + Pseudo-Labeling)** | **0.96799** |

---

##  Project Structure
```text
├── notebooks/
│   └── smartphone-addiction-stacking-ensemble-auc-0-97.ipynb
├── docs/
│   └── project_blog_report.pdf
├── README.md
└── requirements.txt
