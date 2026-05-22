# 💳 Credit Card Fraud Detection

> Machine learning fraud detection system on 283,726 credit card transactions, achieving 0.9703 ROC-AUC with XGBoost and presented through a 4-page Power BI dashboard.

---

## 📌 Problem Statement

Credit card fraud is a critical financial threat — costing billions globally and eroding customer trust. The challenge is severe class imbalance: fraud accounts for only **0.167% of transactions**, meaning naive models can achieve 99.83% accuracy by predicting everything as genuine while catching **zero fraud**.

**Objective:** Build a machine learning pipeline that detects fraudulent transactions with high recall (catching real fraud) while maintaining acceptable precision (avoiding false alarms on genuine customers).

---

## 📊 Dataset Overview

- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Original size:** 284,807 transactions × 31 columns
- **After cleaning:** 283,726 transactions (1,081 duplicates removed)
- **Fraud cases:** 473 (0.167%)
- **Genuine cases:** 283,253 (99.833%)
- **Features:** 28 PCA-transformed (V1–V28), Time, Amount, Class (target)

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.11 |
| **Data Manipulation** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Preprocessing** | StandardScaler, train_test_split |
| **Imbalance Handling** | SMOTE (imbalanced-learn) |
| **Modeling** | Scikit-learn, XGBoost |
| **Evaluation** | classification_report, ROC-AUC, Confusion Matrix |
| **Dashboard** | Power BI (4 pages) |

---

## 🔄 Project Workflow

### Phase 1 — Data Loading & Quality Audit
- Loaded `creditcard.csv` (284,807 rows × 31 columns)
- Removed 1,081 duplicate rows (19 of which were fraud — likely logging errors)
- Confirmed zero missing values
- Verified fraud rate: 0.167%

### Phase 2 — Exploratory Data Analysis (8 Charts)
1. **Class Distribution** — bar + pie chart showing 99.83% vs 0.17% imbalance
2. **Amount Distribution** — histograms for fraud vs genuine
3. **Fraud by Hour** — converted Time to Hour, found peak fraud window
4. **Feature Correlation** — top positive and negative correlators with fraud
5. **Box Plot Analysis** — median amount comparison ($9.82 fraud vs $22.00 genuine)
6. **Confusion Matrix** — comparison across all 3 models
7. **ROC Curve Comparison** — all 3 models on one chart
8. **Feature Importance** — XGBoost top 15 features

### Phase 3 — Preprocessing
- Applied **StandardScaler** to Amount column
- Dropped Time and original Amount columns
- **Stratified train-test split** 80/20 to preserve fraud ratio
- Applied **SMOTE only on training data** to prevent data leakage
  - Before SMOTE: 378 fraud / 226,602 genuine
  - After SMOTE: 226,602 fraud / 226,602 genuine (perfect balance)

### Phase 4 — Model Training & Comparison
Trained 3 models on SMOTE-balanced training data, evaluated on original imbalanced test set:

| Model | AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.9633 | 0.0529 | 0.8737 | 0.0998 |
| Random Forest | 0.9594 | 0.9000 | 0.7579 | 0.8229 |
| **XGBoost ✅** | **0.9703** | **0.7895** | **0.7895** | **0.7895** |

### Phase 5 — Export for Power BI
Exported 5 CSV files for dashboard:
- `transactions_with_predictions.csv`
- `model_comparison.csv`
- `feature_importance.csv`
- `fraud_by_hour.csv`
- `confusion_matrix_data.csv`

### Phase 6 — Power BI Dashboard (4 pages)
1. **Executive Overview** — KPIs, fraud by hour, transaction split, top fraud indicators<img width="1163" height="663" alt="Screenshot 2026-05-22 122225" src="https://github.com/user-attachments/assets/68789661-07fb-4573-b9cc-bbdd7758a1b3" />

2. **Transaction Analysis** — avg amount, fraud by hour line chart, top fraud transactions table <img width="1160" height="654" alt="Screenshot 2026-05-22 122239" src="https://github.com/user-attachments/assets/0943523a-52e2-4697-96d5-b7045ad5b7b1" />

3. **Model Performance** — AUC comparison, precision vs recall, confusion matrices <img width="1160" height="649" alt="Screenshot 2026-05-22 122256" src="https://github.com/user-attachments/assets/94e66398-ac9e-48df-a803-6bf96cad897a" />

4. **Feature Intelligence** — V14 primary signal, secondary signals breakdown <img width="1165" height="655" alt="Screenshot 2026-05-22 122311" src="https://github.com/user-attachments/assets/26ef208e-c53d-4b9a-88f9-9c9302966bbf" />


---

## 🔍 Key Findings & Insights

### Finding 1 — Extreme Class Imbalance
- Only 473 fraud cases out of 283,726 transactions (0.167%)
- Accuracy is misleading here — recall and AUC are the right metrics

### Finding 2 — Fraud Transactions Are Higher Value
- Genuine median amount: **$22.00**
- Fraud median amount: **$9.82** (testing pattern)
- Fraud max amount: **$2,125** (testing then spiking)
- Average fraud transaction: $123.87 vs $88.41 genuine (**40% higher**)

### Finding 3 — Time-Based Fraud Pattern
- Peak fraud hour: **2 AM** with **1.45% fraud rate**
- Off-peak fraud rate stays below 0.32%
- Risk teams should apply stricter checks during 1–3 AM window

### Finding 4 — V14 Dominates Detection
- **V14 alone accounts for 61.52% of XGBoost feature importance**
- V4 (5.44%), V8 (3.31%), V12 (2.58%) are secondary signals
- Top 4 features combined drive 75% of detection power

### Finding 5 — XGBoost Wins on Balance
- **75 fraud caught** out of 95 in test set
- Only **20 false positives** (genuine customers wrongly flagged)
- Random Forest had higher precision but missed more fraud
- Logistic Regression caught more fraud but generated 1,486 false alarms

---

## ⚠️ Challenges & Solutions

| Challenge | Solution |
|---|---|
| **Severe class imbalance (0.167% fraud)** | Used SMOTE on training data only, scale_pos_weight in XGBoost |
| **Data leakage risk if SMOTE applied before split** | Applied SMOTE AFTER train-test split, never on test data |
| **Choosing the right metric (accuracy is misleading)** | Focused on ROC-AUC and recall instead of accuracy |
| **Anonymous PCA features (V1-V28) hard to interpret** | Used XGBoost feature importance to identify V14 as primary signal |
| **1,081 duplicate transactions** | Removed duplicates before training to prevent memorization |
| **Real-time business actionability** | Exported feature importance + hourly patterns to Power BI for ops team |

---

## 💼 Business Recommendations

1. **Implement real-time V14 threshold monitoring** in payment gateway — single feature catches 61.5% of fraud
2. **Apply stricter fraud rules during 1–3 AM** — fraud rate is 4.5x higher than average during this window
3. **Flag transactions where V14, V4, V8, V12 deviate sharply** — top 4 features combined catch 75% of fraud
4. **Use risk scoring instead of binary block** — leverage XGBoost predict_proba for tiered intervention
5. **Set false alarm budget at 20 per period** — XGBoost balances catching 75 frauds against only 20 false positives

---

## 🚀 Future Scope

- Deploy as REST API for real-time scoring (Flask/FastAPI)
- Implement SHAP values for individual transaction explanations
- Add LSTM/sequential models for transaction velocity patterns
- A/B test threshold values to balance precision vs recall in production
- Build automated retraining pipeline with concept drift detection
- Integrate with case management system for fraud investigator workflow

---

## 📂 Project Structure

```
Credit Card Fraud Detection/
├── Credit_Card_Fraud_Detection.ipynb    # Full notebook
├── creditcard.csv                       # Original Kaggle dataset
├── transactions_with_predictions.csv    # Export for Power BI
├── model_comparison.csv                 # Model metrics
├── feature_importance.csv               # XGBoost importance
├── fraud_by_hour.csv                    # Hourly patterns
├── confusion_matrix_data.csv            # CM for all models
├── charts/                              # 8 EDA chart PNGs
└── FRAUD_DETECTION.pbix                 # 4-page Power BI dashboard
```

---

## 📬 About / Contact

**Hari Ram** — Data Analyst | Hyderabad
B.Tech CSE, CVR College of Engineering (2024) | AI Variant Intern

- 📧 **Email:** ramh2125@gmail.com
- 📍 **Location:** Hyderabad, India
- 💼 **Open to:** Data Analyst / Junior Data Scientist roles

