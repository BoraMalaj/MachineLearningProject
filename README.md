# Stroke Prediction — Machine Learning Classification Project
 
A complete machine learning classification project using the **Stroke Prediction Dataset**, organized into two phases: EDA & Preprocessing, and Classification Modeling.
 
---
 
## Project Structure
 
```
├── data/
│   └── healthcare-dataset-stroke-data.csv
├── notebooks/
│   ├── phase1_eda_preprocessing.ipynb
│   └── phase2_modeling.ipynb
├── report/
│   └── stroke_prediction_report.pdf
└── README.md
```
 
---
 
## Phase 1 — EDA & Preprocessing
 
### Problem Definition
The goal is to predict whether a patient is likely to suffer a stroke based on clinical and demographic features such as age, BMI, glucose level, hypertension, and smoking status.
 
### Dataset
- **Source:** Kaggle — [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- **Samples:** 5,110 records
- **Features:** 11 input features + 1 target (`stroke`)
- **Class distribution:** Highly imbalanced (~4.9% stroke cases)
 
### EDA Highlights
- Distribution analysis of numerical features (age, BMI, avg_glucose_level)
- Class imbalance visualization
- Correlation analysis between features and stroke outcome
- Categorical feature analysis (work type, smoking status, residence type)
 
### Preprocessing Steps
- Handling missing values (BMI column imputed with median)
- Encoding categorical variables using One-Hot Encoding
- Feature scaling with StandardScaler
- Addressing class imbalance using **SMOTE**
- Train/test split: **80% train — 20% test** (stratified)
 
---
 
## Phase 2 — Classification Modeling
 
### Models Implemented
 
| Model | Training Time |
|-------|--------------|
| Logistic Regression (Baseline) | 17.9s |
| SVM | 139.7s |
| XGBoost | 46.3s |
 
### Evaluation Metrics
All models were evaluated using:
- Accuracy
- Precision, Recall, F1-Score
- ROC-AUC Score
- Confusion Matrix
 
### Key Results
 
**Logistic Regression:**
- Dominant features: work type and smoking status (one-hot encoded), age
- Fast training, interpretable coefficients
 
**SVM:**
- 830 correct No Stroke, 18 correct Stroke predictions
- Struggles with minority class (32 false negatives)
- Slowest training time (139.7s)
 
**XGBoost:**
- Age is the most important feature (score ≈ 0.175)
- Best balance between performance and training time
- Captures non-linear relationships effectively
 
### Feature Importance Summary
- **Age** is the strongest clinical predictor across all models
- **Hypertension**, **heart disease**, and **ever_married** are consistently important
- **Residence type** and **avg_glucose_level** contribute least
 
---
 
## How to Run
 
```bash
# Clone the repository
git clone https://github.com/your-username/stroke-prediction.git
cd stroke-prediction
 
# Install dependencies
pip install -r requirements.txt
 
# Run notebooks in order
jupyter notebook notebooks/phase1_eda_preprocessing.ipynb
jupyter notebook notebooks/phase2_modeling.ipynb
```
 
### Requirements
```
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
matplotlib
seaborn
jupyter
```
 
---
 
## Authors
- Bora Malaj
- Markela Mykaj
