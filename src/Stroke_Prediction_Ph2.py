# ============================================================
#   STROKE PREDICTION — PHASE 2: CLASSIFICATION MODELS
#   Dataset: Kaggle Stroke Prediction Dataset
#   Models: Logistic Regression | SVM | XGBoost
#   Authors: Bora Malaj, Markela Mykaj
#   Course: Machine Learning — UNYT, 2026
# ============================================================

# ── Core Libraries ──────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import time
warnings.filterwarnings('ignore')

# ── Preprocessing (reuse Phase 1 setup) ─────────────────────
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score, GridSearchCV)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# ── Models ───────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

# ── Evaluation ───────────────────────────────────────────────
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, classification_report,
                             ConfusionMatrixDisplay)

# ── Class Imbalance ──────────────────────────────────────────
from imblearn.over_sampling import SMOTE

# ── Display Settings ─────────────────────────────────────────
pd.set_option('display.max_columns', None)
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
sns.set_style('whitegrid')
sns.set_palette('Set2')

print('✅ All libraries imported successfully!')


# ============================================================
#   SECTION 0 — RELOAD & PREPARE DATA (Same as Phase 1)
# ============================================================

df = pd.read_csv('data/healthcare-dataset-stroke-data.csv')

# --- Cleaning (mirror Phase 1) ---
df['bmi'] = df['bmi'].fillna(df['bmi'].median())
df.drop(columns=['id'], inplace=True)
df = df[df['gender'] != 'Other']

# --- Encoding ---
le = LabelEncoder()
for col in ['gender', 'ever_married', 'Residence_type']:
    df[col] = le.fit_transform(df[col])

df = pd.get_dummies(df, columns=['work_type', 'smoking_status'], drop_first=False)

# --- Features & Target ---
target = 'stroke'
numerical_features = ['age', 'avg_glucose_level', 'bmi']

X = df.drop(columns=[target])
y = df[target]

# --- Scaling ---
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[numerical_features] = scaler.fit_transform(X[numerical_features])

# --- Stratified Train/Test Split (80/20) ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# --- SMOTE on Training Set Only ---
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# --- Cross-Validation Strategy ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f'   Data ready.')
print(f'   Train (after SMOTE): {X_train_smote.shape}')
print(f'   Test (original):     {X_test.shape}')
print(f'   Features:            {X_train_smote.shape[1]}')


# ============================================================
#   SECTION 1 — LOGISTIC REGRESSION (Baseline)
# ============================================================

print('\n' + '='*60)
print('  MODEL 1: LOGISTIC REGRESSION (Baseline)')
print('='*60)

# --- Hyperparameter Tuning ---
lr_param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['lbfgs', 'liblinear'],
    'max_iter': [500, 1000]
}

lr_base = LogisticRegression(random_state=42)
lr_grid = GridSearchCV(lr_base, lr_param_grid, cv=cv,
                       scoring='roc_auc', n_jobs=-1, verbose=0)

start = time.time()
lr_grid.fit(X_train_smote, y_train_smote)
lr_train_time = time.time() - start

best_lr = lr_grid.best_estimator_
print(f'Best Parameters: {lr_grid.best_params_}')
print(f'Best CV ROC-AUC: {lr_grid.best_score_:.4f}')
print(f'Training Time:   {lr_train_time:.2f}s')

# --- Cross-Validation Scores ---
lr_cv_acc  = cross_val_score(best_lr, X_train_smote, y_train_smote, cv=cv, scoring='accuracy')
lr_cv_f1   = cross_val_score(best_lr, X_train_smote, y_train_smote, cv=cv, scoring='f1')
lr_cv_auc  = cross_val_score(best_lr, X_train_smote, y_train_smote, cv=cv, scoring='roc_auc')

print(f'\n5-Fold CV Results:')
print(f'  Accuracy:  {lr_cv_acc.mean():.4f} ± {lr_cv_acc.std():.4f}')
print(f'  F1-Score:  {lr_cv_f1.mean():.4f} ± {lr_cv_f1.std():.4f}')
print(f'  ROC-AUC:   {lr_cv_auc.mean():.4f} ± {lr_cv_auc.std():.4f}')

# --- Test Set Evaluation ---
y_pred_lr   = best_lr.predict(X_test)
y_proba_lr  = best_lr.predict_proba(X_test)[:, 1]

lr_results = {
    'Accuracy':  accuracy_score(y_test, y_pred_lr),
    'Precision': precision_score(y_test, y_pred_lr),
    'Recall':    recall_score(y_test, y_pred_lr),
    'F1-Score':  f1_score(y_test, y_pred_lr),
    'ROC-AUC':   roc_auc_score(y_test, y_proba_lr),
    'Train Time': lr_train_time
}

print('\nTest Set Results:')
for k, v in lr_results.items():
    print(f'  {k}: {v:.4f}' if k != 'Train Time' else f'  {k}: {v:.2f}s')

print('\nClassification Report:')
print(classification_report(y_test, y_pred_lr, target_names=['No Stroke', 'Stroke']))


# ── Plot: LR Confusion Matrix ────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_lr,
    display_labels=['No Stroke', 'Stroke'],
    cmap='Blues', ax=ax
)
ax.set_title('Logistic Regression — Confusion Matrix', fontweight='bold')
plt.tight_layout()
plt.savefig('lr_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Plot: LR Feature Importance (Coefficients) ───────────────
coef_df = pd.DataFrame({
    'Feature': X_train_smote.columns,
    'Coefficient': np.abs(best_lr.coef_[0])
}).sort_values('Coefficient', ascending=False).head(15)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=coef_df, x='Coefficient', y='Feature', palette='Blues_r', ax=ax)
ax.set_title('Logistic Regression — Top 15 Feature Importances (|Coefficients|)',
             fontweight='bold')
ax.set_xlabel('Absolute Coefficient Value')
plt.tight_layout()
plt.savefig('lr_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()


# ============================================================
#   SECTION 2 — SUPPORT VECTOR MACHINE (SVM)
# ============================================================

print('\n' + '='*60)
print('  MODEL 2: SUPPORT VECTOR MACHINE (SVM)')
print('='*60)

# --- Hyperparameter Tuning ---
svm_param_grid = {
    'C':      [0.1, 1, 10],
    'kernel': ['rbf', 'linear'],
    'gamma':  ['scale', 'auto']
}

svm_base = SVC(probability=True, random_state=42)
svm_grid = GridSearchCV(svm_base, svm_param_grid, cv=cv,
                        scoring='roc_auc', n_jobs=-1, verbose=0)

start = time.time()
svm_grid.fit(X_train_smote, y_train_smote)
svm_train_time = time.time() - start

best_svm = svm_grid.best_estimator_
print(f'Best Parameters: {svm_grid.best_params_}')
print(f'Best CV ROC-AUC: {svm_grid.best_score_:.4f}')
print(f'Training Time:   {svm_train_time:.2f}s')

# --- Cross-Validation Scores ---
svm_cv_acc = cross_val_score(best_svm, X_train_smote, y_train_smote, cv=cv, scoring='accuracy')
svm_cv_f1  = cross_val_score(best_svm, X_train_smote, y_train_smote, cv=cv, scoring='f1')
svm_cv_auc = cross_val_score(best_svm, X_train_smote, y_train_smote, cv=cv, scoring='roc_auc')

print(f'\n5-Fold CV Results:')
print(f'  Accuracy:  {svm_cv_acc.mean():.4f} ± {svm_cv_acc.std():.4f}')
print(f'  F1-Score:  {svm_cv_f1.mean():.4f} ± {svm_cv_f1.std():.4f}')
print(f'  ROC-AUC:   {svm_cv_auc.mean():.4f} ± {svm_cv_auc.std():.4f}')

# --- Test Set Evaluation ---
y_pred_svm  = best_svm.predict(X_test)
y_proba_svm = best_svm.predict_proba(X_test)[:, 1]

svm_results = {
    'Accuracy':  accuracy_score(y_test, y_pred_svm),
    'Precision': precision_score(y_test, y_pred_svm),
    'Recall':    recall_score(y_test, y_pred_svm),
    'F1-Score':  f1_score(y_test, y_pred_svm),
    'ROC-AUC':   roc_auc_score(y_test, y_proba_svm),
    'Train Time': svm_train_time
}

print('\nTest Set Results:')
for k, v in svm_results.items():
    print(f'  {k}: {v:.4f}' if k != 'Train Time' else f'  {k}: {v:.2f}s')

print('\nClassification Report:')
print(classification_report(y_test, y_pred_svm, target_names=['No Stroke', 'Stroke']))


# ── Plot: SVM Confusion Matrix ───────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_svm,
    display_labels=['No Stroke', 'Stroke'],
    cmap='Oranges', ax=ax
)
ax.set_title('SVM — Confusion Matrix', fontweight='bold')
plt.tight_layout()
plt.savefig('svm_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()


# ============================================================
#   SECTION 3 — XGBOOST
# ============================================================

print('\n' + '='*60)
print('  MODEL 3: XGBOOST (Gradient Boosting)')
print('='*60)

# Calculate scale_pos_weight for XGBoost (ratio of negative/positive in ORIGINAL train)
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
print(f'scale_pos_weight (original imbalance ratio): {scale_pos:.2f}')
print('Note: Using SMOTE-balanced data → scale_pos_weight=1 for fair comparison')

# --- Hyperparameter Tuning ---
xgb_param_grid = {
    'n_estimators':    [100, 200, 300],
    'max_depth':       [3, 5, 7],
    'learning_rate':   [0.01, 0.05, 0.1],
    'subsample':       [0.8, 1.0],
    'colsample_bytree':[0.8, 1.0]
}

xgb_base = XGBClassifier(
    eval_metric='logloss',
    random_state=42,
    use_label_encoder=False
)

xgb_grid = GridSearchCV(xgb_base, xgb_param_grid, cv=cv,
                        scoring='roc_auc', n_jobs=-1, verbose=0)

start = time.time()
xgb_grid.fit(X_train_smote, y_train_smote)
xgb_train_time = time.time() - start

best_xgb = xgb_grid.best_estimator_
print(f'Best Parameters: {xgb_grid.best_params_}')
print(f'Best CV ROC-AUC: {xgb_grid.best_score_:.4f}')
print(f'Training Time:   {xgb_train_time:.2f}s')

# --- Cross-Validation Scores ---
xgb_cv_acc = cross_val_score(best_xgb, X_train_smote, y_train_smote, cv=cv, scoring='accuracy')
xgb_cv_f1  = cross_val_score(best_xgb, X_train_smote, y_train_smote, cv=cv, scoring='f1')
xgb_cv_auc = cross_val_score(best_xgb, X_train_smote, y_train_smote, cv=cv, scoring='roc_auc')

print(f'\n5-Fold CV Results:')
print(f'  Accuracy:  {xgb_cv_acc.mean():.4f} ± {xgb_cv_acc.std():.4f}')
print(f'  F1-Score:  {xgb_cv_f1.mean():.4f} ± {xgb_cv_f1.std():.4f}')
print(f'  ROC-AUC:   {xgb_cv_auc.mean():.4f} ± {xgb_cv_auc.std():.4f}')

# --- Test Set Evaluation ---
y_pred_xgb  = best_xgb.predict(X_test)
y_proba_xgb = best_xgb.predict_proba(X_test)[:, 1]

xgb_results = {
    'Accuracy':  accuracy_score(y_test, y_pred_xgb),
    'Precision': precision_score(y_test, y_pred_xgb),
    'Recall':    recall_score(y_test, y_pred_xgb),
    'F1-Score':  f1_score(y_test, y_pred_xgb),
    'ROC-AUC':   roc_auc_score(y_test, y_proba_xgb),
    'Train Time': xgb_train_time
}

print('\nTest Set Results:')
for k, v in xgb_results.items():
    print(f'  {k}: {v:.4f}' if k != 'Train Time' else f'  {k}: {v:.2f}s')

print('\nClassification Report:')
print(classification_report(y_test, y_pred_xgb, target_names=['No Stroke', 'Stroke']))


# ── Plot: XGBoost Confusion Matrix ───────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_xgb,
    display_labels=['No Stroke', 'Stroke'],
    cmap='Greens', ax=ax
)
ax.set_title('XGBoost — Confusion Matrix', fontweight='bold')
plt.tight_layout()
plt.savefig('xgb_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Plot: XGBoost Feature Importance ────────────────────────
xgb_feat_imp = pd.DataFrame({
    'Feature':    X_train_smote.columns,
    'Importance': best_xgb.feature_importances_
}).sort_values('Importance', ascending=False).head(15)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=xgb_feat_imp, x='Importance', y='Feature', palette='Greens_r', ax=ax)
ax.set_title('XGBoost — Top 15 Feature Importances', fontweight='bold')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig('xgb_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()


# ============================================================
#   SECTION 4 — MODEL COMPARISON & FINAL EVALUATION
# ============================================================

print('\n' + '='*60)
print('  SECTION 4: MODEL COMPARISON')
print('='*60)

# ── 4.1 Summary Table ────────────────────────────────────────
results_df = pd.DataFrame({
    'Model':     ['Logistic Regression', 'SVM', 'XGBoost'],
    'Accuracy':  [lr_results['Accuracy'],  svm_results['Accuracy'],  xgb_results['Accuracy']],
    'Precision': [lr_results['Precision'], svm_results['Precision'], xgb_results['Precision']],
    'Recall':    [lr_results['Recall'],    svm_results['Recall'],    xgb_results['Recall']],
    'F1-Score':  [lr_results['F1-Score'],  svm_results['F1-Score'],  xgb_results['F1-Score']],
    'ROC-AUC':   [lr_results['ROC-AUC'],   svm_results['ROC-AUC'],   xgb_results['ROC-AUC']],
    'Train Time (s)': [lr_results['Train Time'], svm_results['Train Time'], xgb_results['Train Time']]
}).round(4)

results_df['Train Time (s)'] = results_df['Train Time (s)'].round(2)
print('\n📊 Final Results Table:')
print(results_df.to_string(index=False))

# ── 4.2 Cross-Validation Summary ────────────────────────────
cv_summary = pd.DataFrame({
    'Model':     ['Logistic Regression', 'SVM', 'XGBoost'],
    'CV Accuracy': [f'{lr_cv_acc.mean():.4f} ± {lr_cv_acc.std():.4f}',
                    f'{svm_cv_acc.mean():.4f} ± {svm_cv_acc.std():.4f}',
                    f'{xgb_cv_acc.mean():.4f} ± {xgb_cv_acc.std():.4f}'],
    'CV F1-Score': [f'{lr_cv_f1.mean():.4f} ± {lr_cv_f1.std():.4f}',
                    f'{svm_cv_f1.mean():.4f} ± {svm_cv_f1.std():.4f}',
                    f'{xgb_cv_f1.mean():.4f} ± {xgb_cv_f1.std():.4f}'],
    'CV ROC-AUC':  [f'{lr_cv_auc.mean():.4f} ± {lr_cv_auc.std():.4f}',
                    f'{svm_cv_auc.mean():.4f} ± {svm_cv_auc.std():.4f}',
                    f'{xgb_cv_auc.mean():.4f} ± {xgb_cv_auc.std():.4f}']
})
print('\n📊 Cross-Validation Summary (5-Fold):')
print(cv_summary.to_string(index=False))


# ── 4.3 ROC Curves (All 3 Models) ───────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))

for name, y_proba, color in [
    ('Logistic Regression', y_proba_lr,  'steelblue'),
    ('SVM',                 y_proba_svm, 'darkorange'),
    ('XGBoost',             y_proba_xgb, 'mediumseagreen')
]:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    ax.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})', color=color, lw=2)

ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
ax.fill_between([0, 1], [0, 1], alpha=0.05, color='gray')
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title('ROC Curves — All Models Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
plt.tight_layout()
plt.savefig('roc_curves_comparison.png', dpi=150, bbox_inches='tight')
plt.show()


# ── 4.4 Confusion Matrices Side by Side ─────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
cmaps = ['Blues', 'Oranges', 'Greens']
models_info = [
    ('Logistic Regression', y_pred_lr),
    ('SVM',                 y_pred_svm),
    ('XGBoost',             y_pred_xgb)
]

for ax, (name, y_pred), cmap in zip(axes, models_info, cmaps):
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=['No Stroke', 'Stroke'],
        cmap=cmap, ax=ax
    )
    ax.set_title(f'{name}', fontweight='bold', fontsize=13)

plt.suptitle('Confusion Matrices — All Models', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('all_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()


# ── 4.5 Metrics Bar Chart Comparison ────────────────────────
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x = np.arange(len(metrics))
width = 0.25

fig, ax = plt.subplots(figsize=(13, 6))
bars1 = ax.bar(x - width, [lr_results[m]  for m in metrics], width,
               label='Logistic Regression', color='steelblue',    edgecolor='black', alpha=0.9)
bars2 = ax.bar(x,          [svm_results[m] for m in metrics], width,
               label='SVM',                color='darkorange',    edgecolor='black', alpha=0.9)
bars3 = ax.bar(x + width,  [xgb_results[m] for m in metrics], width,
               label='XGBoost',            color='mediumseagreen', edgecolor='black', alpha=0.9)

ax.set_xlabel('Metric', fontsize=13)
ax.set_ylabel('Score', fontsize=13)
ax.set_title('Model Performance Comparison — All Metrics', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=12)
ax.set_ylim(0, 1.15)
ax.legend(fontsize=11)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('metrics_comparison.png', dpi=150, bbox_inches='tight')
plt.show()


# ── 4.6 CV Scores Comparison Plot ───────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
cv_data = {
    'Accuracy': [lr_cv_acc,  svm_cv_acc,  xgb_cv_acc],
    'F1-Score': [lr_cv_f1,   svm_cv_f1,   xgb_cv_f1],
    'ROC-AUC':  [lr_cv_auc,  svm_cv_auc,  xgb_cv_auc]
}
colors = ['steelblue', 'darkorange', 'mediumseagreen']
model_names = ['LR', 'SVM', 'XGB']

for ax, (metric, scores_list) in zip(axes, cv_data.items()):
    means = [s.mean() for s in scores_list]
    stds  = [s.std()  for s in scores_list]
    bars  = ax.bar(model_names, means, color=colors, edgecolor='black',
                   yerr=stds, capsize=6, alpha=0.9)
    ax.set_title(f'CV {metric}', fontweight='bold', fontsize=13)
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.15)
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 0.02,
                f'{mean:.3f}', ha='center', fontsize=10, fontweight='bold')

plt.suptitle('5-Fold Cross-Validation Scores — Model Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('cv_comparison.png', dpi=150, bbox_inches='tight')
plt.show()


# ── 4.7 Training Time Comparison ─────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
times  = [lr_results['Train Time'], svm_results['Train Time'], xgb_results['Train Time']]
colors = ['steelblue', 'darkorange', 'mediumseagreen']
bars   = ax.bar(['Logistic\nRegression', 'SVM', 'XGBoost'], times,
                color=colors, edgecolor='black', alpha=0.9)
ax.set_title('Training Time Comparison', fontweight='bold')
ax.set_ylabel('Time (seconds)')
for bar, t in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
            f'{t:.1f}s', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('training_time_comparison.png', dpi=150, bbox_inches='tight')
plt.show()


# ============================================================
#   SECTION 5 — FINAL SUMMARY PRINT
# ============================================================

print('\n' + '='*60)
print('       PHASE 2 — FINAL SUMMARY')
print('='*60)

best_model_name = results_df.loc[results_df['ROC-AUC'].idxmax(), 'Model']
best_auc        = results_df['ROC-AUC'].max()
best_f1         = results_df.loc[results_df['F1-Score'].idxmax(), 'Model']

print(f'\n✅ Best ROC-AUC:  {best_model_name}  ({best_auc:.4f})')
print(f'✅ Best F1-Score: {best_f1}')
print(f'\n📊 Final Results Table:')
print(results_df.to_string(index=False))

print(f'\n📁 Saved Figures:')
figures = [
    'lr_confusion_matrix.png',   'lr_feature_importance.png',
    'svm_confusion_matrix.png',
    'xgb_confusion_matrix.png',  'xgb_feature_importance.png',
    'roc_curves_comparison.png', 'all_confusion_matrices.png',
    'metrics_comparison.png',    'cv_comparison.png',
    'training_time_comparison.png'
]
for f in figures:
    print(f'   {f}')

print('\n' + '='*60)
print('  Phase 2 Complete! Ready for Report Paper.')
print('='*60)