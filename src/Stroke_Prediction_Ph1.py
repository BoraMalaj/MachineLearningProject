# Core libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Class imbalance
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Metrics preview
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


# Display settings
pd.set_option('display.max_columns', None)
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
sns.set_style('whitegrid')
sns.set_palette('Set2')

print('All libraries imported successfully!')


# Load the dataset
# Download from: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
df = pd.read_csv('data/healthcare-dataset-stroke-data.csv')

print(f'Dataset Shape: {df.shape}')
print(f'\nColumns: {list(df.columns)}')
df.head()


# Basic info
print('=== Dataset Info ===')
df.info()
print('\n=== Data Types ===')
print(df.dtypes)


# Check missing values
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Missing Percentage (%)': missing_pct.round(2)
}).sort_values('Missing Count', ascending=False)

print('=== Missing Values Summary ===')
print(missing_df[missing_df['Missing Count'] > 0])


# Visualize missing values
fig, ax = plt.subplots(figsize=(10, 5))
missing_plot = missing_pct[missing_pct > 0]
if len(missing_plot) > 0:
    missing_plot.plot(kind='bar', color='coral', edgecolor='black', ax=ax)
    ax.set_title('Missing Values per Feature (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Missing Percentage (%)')
    ax.set_xlabel('Features')
    plt.xticks(rotation=45)
    for i, v in enumerate(missing_plot):
        ax.text(i, v + 0.1, f'{v:.2f}%', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig('missing_values.png', dpi=150, bbox_inches='tight')
    plt.show()
else:
    print('No missing values found!')


# Handle missing values
# 'bmi' column has ~3.9% missing values → impute with median (robust to outliers)
df['bmi'] = df['bmi'].fillna(df['bmi'].median())

# Drop 'id' column — not a feature
df.drop(columns=['id'], inplace=True)

# Verify no missing values remain
print(f'Missing values after imputation: {df.isnull().sum().sum()}')
print(f'Dataset shape after cleaning: {df.shape}')


# Define feature types
target = 'stroke'

numerical_features = ['age', 'avg_glucose_level', 'bmi']

categorical_features = [
    'gender', 'hypertension', 'heart_disease',
    'ever_married', 'work_type', 'Residence_type', 'smoking_status'
]

print('=== Feature Classification ===')
print(f'Target variable: {target}')
print(f'Numerical features ({len(numerical_features)}): {numerical_features}')
print(f'Categorical features ({len(categorical_features)}): {categorical_features}')

# Summary statistics for numerical features
print('\n=== Numerical Features Statistics ===')
df[numerical_features].describe().round(3)


# Inspect categorical feature distributions
print('=== Categorical Feature Value Counts ===')
for col in categorical_features:
    print(f'\n{col}:')
    print(df[col].value_counts())


# Remove 'Other' gender — only 1 record, not statistically meaningful
df = df[df['gender'] != 'Other']
print(f'Dataset shape after removing gender=Other: {df.shape}')

# Create a copy for encoding
df_encoded = df.copy()

# Binary encoding for features with 2 unique values
binary_cols = ['gender', 'ever_married', 'Residence_type']
le = LabelEncoder()
for col in binary_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col])
    print(f'{col} encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}')

# One-Hot Encoding for multi-class categorical features
multi_cols = ['work_type', 'smoking_status']
df_encoded = pd.get_dummies(df_encoded, columns=multi_cols, drop_first=False)

# hypertension and heart_disease are already 0/1 — no encoding needed
print(f'\nDataset shape after encoding: {df_encoded.shape}')
print(f'Columns after encoding: {list(df_encoded.columns)}')


# Visualize encoding results
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, col in enumerate(['gender', 'ever_married', 'Residence_type']):
    df_encoded[col].value_counts().plot(kind='bar', ax=axes[i], color=['steelblue', 'coral'], edgecolor='black')
    axes[i].set_title(f'{col} (after encoding)', fontweight='bold')
    axes[i].set_xlabel('Encoded Value')
    axes[i].set_ylabel('Count')
    axes[i].tick_params(axis='x', rotation=0)
plt.suptitle('Binary Encoded Features Distribution', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('encoding_distribution.png', dpi=150, bbox_inches='tight')
plt.show()


# Separate features and target
X = df_encoded.drop(columns=[target])
y = df_encoded[target]

print(f'Features shape: {X.shape}')
print(f'Target shape: {y.shape}')
print(f'Target distribution:\n{y.value_counts()}')


# Visualize distributions BEFORE scaling
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, col in enumerate(numerical_features):
    axes[i].hist(X[col], bins=30, color='steelblue', edgecolor='black', alpha=0.8)
    axes[i].set_title(f'{col} — Before Scaling', fontweight='bold')
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')
plt.suptitle('Numerical Features Distribution Before Scaling', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('before_scaling.png', dpi=150, bbox_inches='tight')
plt.show()


# Apply StandardScaler to numerical features
# StandardScaler: mean=0, std=1 — best for SVM and Logistic Regression
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[numerical_features] = scaler.fit_transform(X[numerical_features])

print('=== Scaling Summary ===')
print(f'Mean after scaling: {X_scaled[numerical_features].mean().round(6).to_dict()}')
print(f'Std after scaling: {X_scaled[numerical_features].std().round(6).to_dict()}')


# Visualize distributions AFTER scaling
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, col in enumerate(numerical_features):
    axes[i].hist(X_scaled[col], bins=30, color='mediumseagreen', edgecolor='black', alpha=0.8)
    axes[i].set_title(f'{col} — After Scaling', fontweight='bold')
    axes[i].set_xlabel('Standardized Value')
    axes[i].set_ylabel('Frequency')
plt.suptitle('Numerical Features Distribution After StandardScaler', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('after_scaling.png', dpi=150, bbox_inches='tight')
plt.show()


# Analyze class imbalance
class_counts = y.value_counts()
class_pct = y.value_counts(normalize=True) * 100

print('=== Class Distribution (Before SMOTE) ===')
print(f'No Stroke (0): {class_counts[0]} samples ({class_pct[0]:.2f}%)')
print(f'Stroke    (1): {class_counts[1]} samples ({class_pct[1]:.2f}%)')
print(f'Imbalance Ratio: {class_counts[0]/class_counts[1]:.1f}:1')


# Visualize class imbalance
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bar chart
class_counts.plot(kind='bar', ax=axes[0], color=['steelblue', 'coral'], edgecolor='black')
axes[0].set_title('Class Distribution Before SMOTE', fontweight='bold')
axes[0].set_xlabel('Stroke (0=No, 1=Yes)')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=0)
for i, v in enumerate(class_counts):
    axes[0].text(i, v + 10, str(v), ha='center', fontweight='bold')

# Pie chart
axes[1].pie(class_counts, labels=['No Stroke', 'Stroke'],
            autopct='%1.2f%%', colors=['steelblue', 'coral'],
            startangle=90, explode=(0, 0.1),
            textprops={'fontsize': 12})
axes[1].set_title('Class Proportion Before SMOTE', fontweight='bold')

plt.suptitle('Class Imbalance Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('class_imbalance.png', dpi=150, bbox_inches='tight')
plt.show()


# Apply SMOTE — Synthetic Minority Over-sampling Technique
# SMOTE generates synthetic samples for the minority class (stroke=1)
# Important: SMOTE is applied ONLY on training data (after split)

# First: do the train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print('=== Train/Test Split (Before SMOTE) ===')
print(f'Training set size: {X_train.shape[0]} samples')
print(f'Test set size:     {X_test.shape[0]} samples')
print(f'Train class distribution:\n{y_train.value_counts()}')
print(f'Test class distribution:\n{y_test.value_counts()}')

# Now apply SMOTE only on training data
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print('\n=== After SMOTE (Training Set Only) ===')
print(f'Training set size after SMOTE: {X_train_smote.shape[0]} samples')
print(f'Class distribution after SMOTE:\n{pd.Series(y_train_smote).value_counts()}')


# Visualize class distribution before vs after SMOTE
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Before SMOTE
before = y_train.value_counts()
axes[0].bar(['No Stroke', 'Stroke'], before.values, color=['steelblue', 'coral'], edgecolor='black')
axes[0].set_title('Training Set — Before SMOTE', fontweight='bold')
axes[0].set_ylabel('Count')
for i, v in enumerate(before.values):
    axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold')

# After SMOTE
after = pd.Series(y_train_smote).value_counts()
axes[1].bar(['No Stroke', 'Stroke'], after.values, color=['mediumseagreen', 'darkorange'], edgecolor='black')
axes[1].set_title('Training Set — After SMOTE', fontweight='bold')
axes[1].set_ylabel('Count')
for i, v in enumerate(after.values):
    axes[1].text(i, v + 5, str(v), ha='center', fontweight='bold')

plt.suptitle('Class Distribution: Before vs After SMOTE', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('smote_comparison.png', dpi=150, bbox_inches='tight')
plt.show()


# Comprehensive split summary
print('=' * 50)
print('       TRAIN / TEST SPLIT SUMMARY')
print('=' * 50)
print(f'Total dataset size:        {len(X_scaled)} samples')
print(f'Training set (80%):        {X_train.shape[0]} samples')
print(f'Test set (20%):            {X_test.shape[0]} samples')
print(f'Training after SMOTE:      {X_train_smote.shape[0]} samples')
print(f'Number of features:        {X_scaled.shape[1]}')
print(f'Stratification:            Yes (preserves class ratio)')
print(f'Random state:              42 (reproducibility)')
print('=' * 50)

# Verify stratification
print(f'\nTrain stroke rate: {y_train.mean()*100:.2f}%')
print(f'Test stroke rate:  {y_test.mean()*100:.2f}%')
print('Stratification confirmed: class ratios are preserved')


# Stratified K-Fold Cross Validation Setup
# Used by ALL models for fair comparison

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print('=== Cross-Validation Configuration ===')
print(f'Strategy:        Stratified K-Fold')
print(f'Number of folds: 5')
print(f'Shuffle:         True')
print(f'Random state:    42')
print(f'Why Stratified:  Preserves class imbalance ratio in each fold')

# Demonstrate fold splits
print('\n=== Fold Breakdown ===')
for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_smote, y_train_smote), 1):
    fold_y_train = pd.Series(y_train_smote).iloc[train_idx]
    fold_y_val = pd.Series(y_train_smote).iloc[val_idx]
    print(f'Fold {fold}: Train={len(train_idx)}, Val={len(val_idx)}, '
          f'Train stroke%={fold_y_train.mean()*100:.1f}%, '
          f'Val stroke%={fold_y_val.mean()*100:.1f}%')


# Quick cross-validation demo with Logistic Regression (baseline)
# Just to validate setup — detailed LR implementation is in Phase 2
lr_demo = LogisticRegression(random_state=42, max_iter=1000)

cv_scores_acc = cross_val_score(lr_demo, X_train_smote, y_train_smote, cv=cv, scoring='accuracy')
cv_scores_f1 = cross_val_score(lr_demo, X_train_smote, y_train_smote, cv=cv, scoring='f1')
cv_scores_roc = cross_val_score(lr_demo, X_train_smote, y_train_smote, cv=cv, scoring='roc_auc')

print('=== Cross-Validation Demo Results (Logistic Regression) ===')
print(f'Accuracy:  {cv_scores_acc.mean():.4f} ± {cv_scores_acc.std():.4f}')
print(f'F1-Score:  {cv_scores_f1.mean():.4f} ± {cv_scores_f1.std():.4f}')
print(f'ROC-AUC:   {cv_scores_roc.mean():.4f} ± {cv_scores_roc.std():.4f}')


# Visualize cross-validation scores
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(1, 6)
width = 0.25

ax.bar(x - width, cv_scores_acc, width, label='Accuracy', color='steelblue', edgecolor='black')
ax.bar(x, cv_scores_f1, width, label='F1-Score', color='coral', edgecolor='black')
ax.bar(x + width, cv_scores_roc, width, label='ROC-AUC', color='mediumseagreen', edgecolor='black')

ax.set_xlabel('Fold')
ax.set_ylabel('Score')
ax.set_title('5-Fold Stratified CV Scores (Logistic Regression Demo)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'Fold {i}' for i in range(1, 6)])
ax.set_ylim(0, 1.1)
ax.legend()
ax.axhline(y=cv_scores_acc.mean(), color='steelblue', linestyle='--', alpha=0.5)
ax.axhline(y=cv_scores_f1.mean(), color='coral', linestyle='--', alpha=0.5)
ax.axhline(y=cv_scores_roc.mean(), color='mediumseagreen', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('cv_scores.png', dpi=150, bbox_inches='tight')
plt.show()


# Build a reusable sklearn Pipeline for all models in Phase 2
# This ensures consistent preprocessing for Student A (LR) and all Phase 2 models

# Numerical pipeline: impute → scale
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline: impute → one-hot encode
from sklearn.preprocessing import OneHotEncoder
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
])

# Combined preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'])
], remainder='passthrough')

print('=== Preprocessing Pipeline Structure ===')
print(preprocessor)


# Full ImbPipeline with SMOTE (for use in Phase 2 models)
# This is the pipeline that Phase 2 models (SVM, XGBoost) should use

# Reload raw data for pipeline demo
df_raw = pd.read_csv('data/healthcare-dataset-stroke-data.csv')
df_raw['bmi'] = df_raw['bmi'].fillna(df_raw['bmi'].median())
df_raw = df_raw[df_raw['gender'] != 'Other']
df_raw.drop(columns=['id'], inplace=True)

X_raw = df_raw.drop(columns=['stroke'])
y_raw = df_raw['stroke']

X_raw_train, X_raw_test, y_raw_train, y_raw_test = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
)

full_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42, k_neighbors=5)),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

# Fit and test pipeline
full_pipeline.fit(X_raw_train, y_raw_train)
y_pred_pipeline = full_pipeline.predict(X_raw_test)

print('=== Pipeline Validation (Logistic Regression) ===')
print(classification_report(y_raw_test, y_pred_pipeline, target_names=['No Stroke', 'Stroke']))


# Save preprocessed datasets for use in Phase 2
import os
os.makedirs('preprocessed_data', exist_ok=True)

# Save as CSV
X_train_smote_df = pd.DataFrame(X_train_smote, columns=X_scaled.columns)
X_test_df = pd.DataFrame(X_test, columns=X_scaled.columns)

X_train_smote_df.to_csv('preprocessed_data/X_train_smote.csv', index=False)
X_test_df.to_csv('preprocessed_data/X_test.csv', index=False)
pd.Series(y_train_smote).to_csv('preprocessed_data/y_train_smote.csv', index=False)
y_test.to_csv('preprocessed_data/y_test.csv', index=False)

print('=== Saved Files ===')
for f in os.listdir('preprocessed_data'):
    print(f'  preprocessed_data/{f}')

print(f'\nX_train_smote shape: {X_train_smote_df.shape}')
print(f'X_test shape:        {X_test_df.shape}')
print('\nPreprocessed data saved successfully!')


print('=' * 60)
print('         PHASE 1 - PREPROCESSING SUMMARY')
print('=' * 60)
print()
print('1. MISSING VALUE HANDLING')
print('   - BMI: 201 missing → imputed with median')
print('   - ID column: dropped (not a feature)')
print('   - Gender=Other: 1 record removed')
print()
print('2. CATEGORICAL ENCODING')
print('   - Binary (LabelEncoder): gender, ever_married, Residence_type')
print('   - Multi-class (OneHotEncoder): work_type, smoking_status')
print('   - Already binary: hypertension, heart_disease')
print()
print('3. FEATURE SCALING')
print('   - StandardScaler on: age, avg_glucose_level, bmi')
print('   - Result: mean≈0, std≈1 for all numerical features')
print()
print('4. CLASS IMBALANCE')
print('   - Original ratio: ~20:1 (no stroke vs stroke)')
print('   - Solution: SMOTE on training data ONLY')
print('   - After SMOTE: balanced 1:1 ratio in training set')
print()
print('5. TRAIN/TEST SPLIT')
print('   - Split: 80% train / 20% test')
print('   - Stratified: Yes')
print('   - Random state: 42')
print()
print('6. CROSS-VALIDATION')
print('   - Strategy: Stratified 5-Fold')
print('   - Applied to ALL models in Phase 2')
print('   - Random state: 42')
print()
print('7. PIPELINE')
print('   - Full sklearn/imblearn Pipeline built')
print('   - Ready for SVM, XGBoost, Logistic Regression')
print('=' * 60)

