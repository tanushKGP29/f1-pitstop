"""
Consolidated model development module.
Contains baseline GroupKFold CV, Optuna hyperparameter tuning,
and OOF ensemble training (LightGBM + CatBoost + XGBoost) with blend optimization.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

from .data_cleaning import F1TargetEncoder, target_encode_with_smoothing

# ---------------------------------------------------------------------------
# Model Parameter Configurations
# ---------------------------------------------------------------------------

def get_lgb_params(optuna_cache_path='optuna_lgbm_pitstop_cache.json'):
    """Load Optuna-tuned LightGBM params or fall back to defaults."""
    # Look for cache in root or in src directory
    path = Path(optuna_cache_path)
    if not path.exists():
        # Fallback to check in parent if run from src
        path = Path(__file__).resolve().parent.parent / optuna_cache_path

    if path.exists():
        try:
            cache = json.loads(path.read_text(encoding='utf-8'))
            params = cache['best_params'].copy()
        except Exception:
            params = {}
    else:
        params = {}

    params.setdefault('num_leaves', 81)
    params.setdefault('max_depth', 10)
    params.setdefault('learning_rate', 0.05)
    params.setdefault('subsample', 0.7)
    params.setdefault('colsample_bytree', 0.5)
    params.setdefault('min_child_samples', 44)
    params.setdefault('scale_pos_weight', 2.5)
    params.update({
        'n_estimators': 800,
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
    })
    return params


def get_cb_params():
    return {
        'iterations': 800,
        'learning_rate': 0.05,
        'depth': 8,
        'l2_leaf_reg': 3.0,
        'random_seed': 42,
        'eval_metric': 'AUC',
        'task_type': 'CPU',
        'verbose': 0,
        'auto_class_weights': 'Balanced',
    }


def get_xgb_params():
    return {
        'n_estimators': 800,
        'learning_rate': 0.05,
        'max_depth': 8,
        'subsample': 0.7,
        'colsample_bytree': 0.5,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'scale_pos_weight': 2.5,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0,
        'early_stopping_rounds': 50,
    }


# ---------------------------------------------------------------------------
# Cross-Validation & Optuna Tuning
# ---------------------------------------------------------------------------

def run_group_kfold_cv(X_train, y_train):
    """Run baseline 5-fold GroupKFold CV to check feature quality."""
    train_df = X_train.copy()
    train_df['race_year_group'] = train_df['Race'] + '_' + train_df['Year'].astype(str) if 'Race' in train_df.columns else 'group'

    gkf = GroupKFold(n_splits=5)
    groups = train_df['race_year_group']

    baseline_params = dict(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    cv_roc, cv_pr = []
    print('Running 5-fold GroupKFold CV...')

    for fold, (tr_idx, cv_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_cv, y_cv = X_train.iloc[cv_idx], y_train.iloc[cv_idx]

        model = lgb.LGBMClassifier(**baseline_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_cv, y_cv)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
        )

        proba = model.predict_proba(X_cv)[:, 1]
        roc = roc_auc_score(y_cv, proba)
        pr = average_precision_score(y_cv, proba)
        cv_roc.append(roc)
        cv_pr.append(pr)
        print(f'  Fold {fold+1}: ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}')

    print(f'\nCV ROC-AUC: {np.mean(cv_roc):.4f} ± {np.std(cv_roc):.4f}')
    print(f'CV PR-AUC:  {np.mean(cv_pr):.4f} ± {np.std(cv_pr):.4f}')
    return {'mean_roc_auc': np.mean(cv_roc), 'mean_pr_auc': np.mean(cv_pr)}


def objective(trial, X_train, y_train, groups):
    """Optuna objective function for tuning LightGBM parameters."""
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 31, 255),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'n_estimators': 800,
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'subsample_freq': 1,
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 150),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.5, 6.0),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.5),
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'random_state': 42,
        'n_jobs': -1,
        'force_col_wise': True,
        'verbose': -1
    }

    gkf_inner = GroupKFold(n_splits=3)
    fold_scores = []
    fold_best_iterations = []

    for tr_idx, cv_idx in gkf_inner.split(X_train, y_train, groups):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_cv, y_cv = X_train.iloc[cv_idx], y_train.iloc[cv_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_cv, y_cv)],
            callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(-1)]
        )

        proba = model.predict_proba(X_cv)[:, 1]
        fold_scores.append(average_precision_score(y_cv, proba))
        best_iter = model.best_iteration_ if model.best_iteration_ is not None else params['n_estimators']
        fold_best_iterations.append(int(best_iter))

    if fold_best_iterations:
        trial.set_user_attr('mean_best_iteration', int(np.mean(fold_best_iterations)))
    return np.mean(fold_scores)


def run_optuna_study(X_train, y_train, groups, cache_path='optuna_lgbm_pitstop_cache.json'):
    """Run Optuna study or load from cache."""
    path = Path(cache_path)
    if not path.is_absolute() and not path.exists():
        path = Path(__file__).resolve().parent.parent / cache_path

    if path.exists():
        try:
            cached = json.loads(path.read_text(encoding='utf-8'))
            print(f'Loaded cached Optuna result from {path.name}')
            return {
                'best_params': cached['best_params'],
                'best_pr_auc': cached['best_pr_auc'],
                'recommended_n_estimators': cached['recommended_n_estimators'],
            }
        except Exception:
            pass

    print('Starting Optuna hyperparameter tuning...')
    study = optuna.create_study(direction='maximize', study_name='lgbm_pitstop', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, X_train, y_train, groups), n_trials=50, show_progress_bar=True)

    print(f'Best PR-AUC: {study.best_value:.4f}')
    
    # Calculate recommended n_estimators
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    if completed_trials:
        top_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:10]
        iterations = [t.user_attrs.get('mean_best_iteration') for t in top_trials]
        iterations = [it for it in iterations if isinstance(it, (int, float)) and it > 0]
        recommended_n_estimators = int(np.clip(np.median(iterations) * 1.15, 200, 1200)) if iterations else 600
    else:
        recommended_n_estimators = 600

    payload = {
        'best_params': study.best_params,
        'best_pr_auc': study.best_value,
        'recommended_n_estimators': recommended_n_estimators,
    }
    try:
        path.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
        print(f'Saved Optuna study cache to {path.name}')
    except Exception as e:
        print(f'Could not save Optuna cache: {e}')

    return payload


# ---------------------------------------------------------------------------
# OOF Ensemble Training & Save
# ---------------------------------------------------------------------------

def train_oof_ensemble(X_trainval, y_trainval, X_test, n_splits=5):
    """
    Train LightGBM, CatBoost, and XGBoost models on 5 Stratified CV folds.
    Returns the out-of-fold predictions, test predictions, and the fitted model instances.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    n_train = len(X_trainval)
    n_test = len(X_test)

    oof = {m: np.zeros(n_train) for m in ('lgb', 'cb', 'xgb')}
    test_preds = {m: np.zeros(n_test) for m in ('lgb', 'cb', 'xgb')}

    lgb_params = get_lgb_params()
    cb_params = get_cb_params()
    xgb_params = get_xgb_params()

    fitted_models = {m: [] for m in ('lgb', 'cb', 'xgb')}

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_trainval, y_trainval)):
        print(f'\n--- Fold {fold + 1}/{n_splits} ({len(tr_idx):,} train / {len(val_idx):,} val) ---')
        X_tr, y_tr = X_trainval.iloc[tr_idx], y_trainval.iloc[tr_idx]
        X_vl, y_vl = X_trainval.iloc[val_idx], y_trainval.iloc[val_idx]

        # --- LightGBM ---
        m_lgb = lgb.LGBMClassifier(**lgb_params)
        m_lgb.fit(
            X_tr, y_tr,
            eval_set=[(X_vl, y_vl)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        oof['lgb'][val_idx] = m_lgb.predict_proba(X_vl)[:, 1]
        test_preds['lgb'] += m_lgb.predict_proba(X_test)[:, 1] / n_splits
        fitted_models['lgb'].append(m_lgb)
        print(f'  LGB  AUC: {roc_auc_score(y_vl, oof["lgb"][val_idx]):.4f}')

        # --- CatBoost ---
        m_cb = cb.CatBoostClassifier(**cb_params)
        m_cb.fit(X_tr, y_tr, eval_set=(X_vl, y_vl), early_stopping_rounds=50, verbose=False)
        oof['cb'][val_idx] = m_cb.predict_proba(X_vl)[:, 1]
        test_preds['cb'] += m_cb.predict_proba(X_test)[:, 1] / n_splits
        fitted_models['cb'].append(m_cb)
        print(f'  CB   AUC: {roc_auc_score(y_vl, oof["cb"][val_idx]):.4f}')

        # --- XGBoost ---
        m_xgb = xgb.XGBClassifier(**xgb_params)
        m_xgb.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
        oof['xgb'][val_idx] = m_xgb.predict_proba(X_vl)[:, 1]
        test_preds['xgb'] += m_xgb.predict_proba(X_test)[:, 1] / n_splits
        fitted_models['xgb'].append(m_xgb)
        print(f'  XGB  AUC: {roc_auc_score(y_vl, oof["xgb"][val_idx]):.4f}')

    return oof, test_preds, fitted_models


def optimize_blend_weights(oof, y_true):
    """Grid search for optimal blend weights on OOF predictions."""
    best_auc, best_w = 0, (1 / 3, 1 / 3, 1 / 3)

    for w1 in np.arange(0.10, 0.80, 0.05):
        for w2 in np.arange(0.10, 0.85 - w1, 0.05):
            w3 = round(1.0 - w1 - w2, 2)
            if w3 < 0.05:
                continue
            blend = w1 * oof['lgb'] + w2 * oof['cb'] + w3 * oof['xgb']
            auc = roc_auc_score(y_true, blend)
            if auc > best_auc:
                best_auc = auc
                best_w = (round(w1, 2), round(w2, 2), round(w3, 2))

    return best_w, best_auc


def optimize_threshold(oof_probs, y_true):
    """Find the threshold that maximizes F1-score on OOF predictions."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, oof_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]
    
    print(f"Optimal OOF threshold: {best_threshold:.4f} (F1: {best_f1:.4f})")
    return best_threshold, best_f1


def run_ensemble(X_trainval, y_trainval, X_test, y_test=None, n_splits=5, models_dir=None):
    """
    Full ensemble training pipeline: OOF training, blend optimization,
    threshold calibration, and test predictions. Optionally serializes models.
    """
    print('=' * 60)
    print('ENSEMBLE TRAINING (OOF)')
    print('=' * 60)

    oof, test_preds, fitted_models = train_oof_ensemble(
        X_trainval, y_trainval, X_test, n_splits
    )

    # --- OOF scores (honest) ---
    print('\n' + '=' * 60)
    print('OOF SCORES')
    print('=' * 60)
    for name, arr in oof.items():
        auc = roc_auc_score(y_trainval, arr)
        ap = average_precision_score(y_trainval, arr)
        print(f'  {name.upper():4s}  OOF ROC-AUC: {auc:.4f},  PR-AUC: {ap:.4f}')

    # --- Blend weights (optimized on OOF only) ---
    weights, oof_blend_auc = optimize_blend_weights(oof, y_trainval)
    print(f'\nOptimal blend weights: LGB={weights[0]:.2f}, CB={weights[1]:.2f}, XGB={weights[2]:.2f}')
    print(f'OOF Blend ROC-AUC: {oof_blend_auc:.4f}')

    # --- Threshold optimization (on OOF blend) ---
    oof_blend = weights[0] * oof['lgb'] + weights[1] * oof['cb'] + weights[2] * oof['xgb']
    best_threshold, best_f1 = optimize_threshold(oof_blend, y_trainval)

    # --- Final test evaluation ---
    test_blend = (
        weights[0] * test_preds['lgb']
        + weights[1] * test_preds['cb']
        + weights[2] * test_preds['xgb']
    )

    if y_test is not None and len(y_test) > 0 and not np.all(y_test == 0):
        test_auc = roc_auc_score(y_test, test_blend)
        test_ap = average_precision_score(y_test, test_blend)
        print(f'\n{"=" * 60}')
        print('FINAL TEST RESULTS')
        print('=' * 60)
        print(f'Test ROC-AUC: {test_auc:.4f}')
        print(f'Test PR-AUC:  {test_ap:.4f}')
        print(f'OOF vs Test gap: {abs(oof_blend_auc - test_auc):.4f}')

    # --- Serialization (Offline Mode Support) ---
    if models_dir is not None:
        models_path = Path(models_dir)
        ensemble_path = models_path / 'ensemble'
        ensemble_path.mkdir(parents=True, exist_ok=True)

        print(f'\nSerializing models and metadata to {models_path.resolve()}...')
        
        # Save models natively
        for i in range(n_splits):
            # LightGBM
            fitted_models['lgb'][i].booster_.save_model(str(ensemble_path / f'lgb_fold_{i}.txt'))
            # CatBoost
            fitted_models['cb'][i].save_model(str(ensemble_path / f'cb_fold_{i}.cbm'), format='cbm')
            # XGBoost
            fitted_models['xgb'][i].save_model(str(ensemble_path / f'xgb_fold_{i}.json'))

        # Save metadata JSON
        metadata = {
            'blend_weights': list(weights),
            'optimal_threshold': float(best_threshold),
            'optimal_f1': float(best_f1),
            'global_mean': float(y_trainval.mean()),
            'feature_names': X_trainval.columns.tolist()
        }
        (models_path / 'ensemble_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        print('Model serialization completed successfully.')

    return test_blend, weights, best_threshold
