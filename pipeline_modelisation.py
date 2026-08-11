# =====================================================================
# Extrait anonymise du pipeline de modelisation (projet churn milieu bancaire)
# Cellules principales : preparation, deux modeles, validation out-of-time.
# Demonstration de la demarche, non executable en l'etat.
# =====================================================================

# ============================================================
# Extraits principaux du notebook Python (Snowflake)
# ============================================================

# --- Imports ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import mannwhitneyu, chi2_contingency, ks_2samp
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (roc_auc_score, confusion_matrix,
                             classification_report, precision_recall_curve)
import lightgbm as lgb
import optuna
import re

# --- Lecture de la table analytique et découpage ---
df = charger_table()   # table analytique construite en SQL (voir sql/construction_table.sql)
cols_bool = df.select_dtypes(include='bool').columns.tolist()
df[cols_bool] = df[cols_bool].astype(int)

Y = df['Y']
X = df.drop(columns=['ID_CLIENT', 'Y'])
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.30, random_state=42, stratify=Y)

# --- Imputation (médiane) sur les écarts-types résiduels ---
cols_imput = ['ECART_TYPE_ENCOURS_12M', 'ECART_TYPE_CREDIT_12M',
              'ECART_TYPE_DEBIT_12M', 'ECART_TYPE_DEBIT_CARTE_12M']
imputer = SimpleImputer(strategy='median')
X_train[cols_imput] = imputer.fit_transform(X_train[cols_imput])
X_test[cols_imput]  = imputer.transform(X_test[cols_imput])

# --- Tests univariés de discrimination ---
cols_quant = [c for c in X_train.columns if X_train[c].nunique() > 20]
cols_bin   = [c for c in X_train.columns if X_train[c].nunique() == 2]

# Mann-Whitney (variables numériques, churn vs non-churn)
res_mw = [{'variable': c,
           'p_value': mannwhitneyu(X_train.loc[y_train == 1, c],
                                   X_train.loc[y_train == 0, c])[1]}
          for c in cols_quant]

# Chi-deux (variables binaires vs Y)  et  Kolmogorov-Smirnov (train vs test)
res_chi2 = [{'variable': c,
             'p_value': chi2_contingency(pd.crosstab(X_train[c], y_train))[1]}
            for c in cols_bin]
res_ks   = [{'variable': c, 'p_value': ks_2samp(X_train[c], X_test[c])[1]}
            for c in cols_quant]

# --- Contrôle de la colinéarité (Pearson) et purge raisonnée ---
mat_corr = X_train[cols_quant].corr()
masque   = np.triu(np.ones(mat_corr.shape), k=1).astype(bool)
paires   = mat_corr.where(masque).stack().reset_index()
paires.columns = ['variable_1', 'variable_2', 'correlation']
paires_fortes = paires[paires['correlation'].abs() > 0.8]
# cols_supp : une variable conservée par famille temporelle (voir rapport)
X_train = X_train.drop(columns=cols_supp)
X_test  = X_test.drop(columns=cols_supp)

# ============================================================
# Modèle 1 : régression logistique (Elastic Net)
# ============================================================
COLS_RATIOS_ZSCORES = [
    'ZSCORE_ENCOURS_M', 'ZSCORE_CREDIT_M', 'ZSCORE_CREDIT_M1',
    'ZSCORE_DEBIT_M', 'ZSCORE_DEBIT_CARTE_M',
    'RATIO_ENCOURS_M_MOY12M', 'RATIO_CREDIT_M_MOY12M',
    'RATIO_DEBIT_M_MOY12M', 'RATIO_DEBIT_CARTE_M_MOY12M']

X_train_std = X_train.copy()
X_test_std  = X_test.copy()
cols_a_scaler = [c for c in cols_quant if c not in COLS_RATIOS_ZSCORES]
scaler = StandardScaler()
X_train_std[cols_a_scaler] = scaler.fit_transform(X_train_std[cols_a_scaler])
X_test_std[cols_a_scaler]  = scaler.transform(X_test_std[cols_a_scaler])

mod_lr = LogisticRegressionCV(
    penalty='elasticnet', l1_ratios=[0.2], Cs=[0.01],
    solver='saga', class_weight='balanced', scoring='roc_auc',
    cv=3, max_iter=3000, random_state=42, n_jobs=-1)
mod_lr.fit(X_train_std, y_train)

y_proba_lr = mod_lr.predict_proba(X_test_std)[:, 1]
y_pred_lr  = mod_lr.predict(X_test_std)
print(f"AUC regression logistique : {roc_auc_score(y_test, y_proba_lr):.4f}")
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# Interprétation : Odds Ratios
coefs_df = pd.DataFrame({'variable': X_train_std.columns,
                         'coefficient': mod_lr.coef_[0]})
coefs_df['odds_ratio'] = np.exp(coefs_df['coefficient'])
coefs_df = coefs_df.sort_values('odds_ratio', ascending=False)

# ============================================================
# Modèle 2 : LightGBM
# ============================================================
# nettoyage des noms de colonnes (LightGBM)
X_train.columns = [re.sub(r'[^A-Za-z0-9]', '_', c) for c in X_train.columns]
X_test.columns  = [re.sub(r'[^A-Za-z0-9]', '_', c) for c in X_test.columns]

mod_lgb = lgb.LGBMClassifier(
    objective='binary', class_weight='balanced',
    n_estimators=300, learning_rate=0.05, num_leaves=31,
    random_state=42, n_jobs=-1, verbosity=-1)
mod_lgb.fit(X_train, y_train)
y_proba_lgb = mod_lgb.predict_proba(X_test)[:, 1]
print(f"AUC LightGBM (defaut) : {roc_auc_score(y_test, y_proba_lgb):.4f}")

# --- Optimisation des hyperparamètres (Optuna) ---
def objective(trial):
    params = {
        'objective': 'binary', 'metric': 'auc', 'verbosity': -1,
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 15),
        'random_state': 42, 'n_jobs': -1}
    modele = lgb.LGBMClassifier(**params)
    return cross_val_score(modele, X_train, y_train,
                           cv=3, scoring='roc_auc').mean()

study = optuna.create_study(
    direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=30)

# --- Modèle final retenu ---
param_finaux = dict(study.best_params)
param_finaux.update({'objective': 'binary', 'random_state': 42,
                     'n_jobs': -1, 'verbosity': -1})
mod_final = lgb.LGBMClassifier(**param_finaux)
mod_final.fit(X_train, y_train)
y_proba_final = mod_final.predict_proba(X_test)[:, 1]
y_pred_final  = mod_final.predict(X_test)
print(f"AUC test : {roc_auc_score(y_test, y_proba_final):.4f}")

# --- Choix du seuil de décision ---
precisions, recalls, seuils = precision_recall_curve(y_test, y_proba_final)
taux_base = y_test.mean()
for s in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    l = tab.iloc[(tab['seuil'] - s).abs().argmin()]
    print(f"Seuil {l['seuil']:.2f} : precision={l['precision']:.1%}, "
          f"recall={l['recall']:.1%}, lift=x{l['precision']/taux_base:.1f}")

# ============================================================
# Validation out-of-time : entraînement 2023 -> test 2024
# ============================================================
X_2023, y_2023 = preparer(df_2023)
X_2024, y_2024 = preparer(df_2024)
X_2024 = X_2024.reindex(columns=X_2023.columns, fill_value=0)

mod_oot = lgb.LGBMClassifier(**param_finaux)
mod_oot.fit(X_2023, y_2023)
y_proba_oot = mod_oot.predict_proba(X_2024)[:, 1]
print(f"AUC out-of-time : {roc_auc_score(y_2024, y_proba_oot):.4f}")
