from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import sys
ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from floodsense.data_pipeline import clean_training_data, attach_elevation_features, get_feature_target_frame, load_raw_data
from floodsense.modeling import build_candidate_pipelines


def metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }


def main():
    train_df, elevation_df, ndma_df = load_raw_data(ROOT)
    cleaned_df, stats = clean_training_data(train_df)
    model_df = attach_elevation_features(cleaned_df, elevation_df)
    X, y = get_feature_target_frame(model_df)

    # Split stratified
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train original candidate (random_forest)
    candidates = build_candidate_pipelines(random_state=42)
    model_full = candidates['random_forest']
    model_full.fit(X_train, y_train)
    y_pred_full = model_full.predict(X_test)
    full_metrics = metrics(y_test, y_pred_full)

    # Remove water area features
    water_cols = [c for c in X.columns if 'water_area' in c]
    Xn_train = X_train.drop(columns=water_cols)
    Xn_test = X_test.drop(columns=water_cols)

    # Need to rebuild pipeline to match columns: we'll create a simple RF with median imputer for numeric only
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    numeric_cols = [c for c in Xn_train.columns if Xn_train[c].dtype.kind in 'fi']
    categorical_cols = [c for c in Xn_train.columns if c not in numeric_cols]

    numeric_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median'))])
    categorical_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                                     ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preproc = ColumnTransformer([('num', numeric_pipeline, numeric_cols), ('cat', categorical_pipeline, categorical_cols)])
    from sklearn.ensemble import RandomForestClassifier
    pipe_no_water = Pipeline([('preproc', preproc), ('clf', RandomForestClassifier(n_estimators=200, random_state=42))])
    pipe_no_water.fit(Xn_train, y_train)
    y_pred_no_water = pipe_no_water.predict(Xn_test)
    no_water_metrics = metrics(y_test, y_pred_no_water)

    # Permutation importance on full model for water features
    from sklearn.inspection import permutation_importance
    result = permutation_importance(model_full, X_test, y_test, n_repeats=10, random_state=42, n_jobs=1)
    importances = dict(zip(X_test.columns, result.importances_mean.tolist()))

    # Ablation experiments
    ablations = {}
    # 1) remove water cols (already computed)
    ablations['no_water'] = no_water_metrics
    # 2) remove ds_idx
    if 'ds_idx' in X.columns:
        Xn2_train = X_train.drop(columns=['ds_idx'])
        Xn2_test = X_test.drop(columns=['ds_idx'])
        numeric_cols = [c for c in Xn2_train.columns if Xn2_train[c].dtype.kind in 'fi']
        categorical_cols = [c for c in Xn2_train.columns if c not in numeric_cols]
        preproc2 = ColumnTransformer([('num', numeric_pipeline, numeric_cols), ('cat', categorical_pipeline, categorical_cols)])
        pipe_ds = Pipeline([('preproc', preproc2), ('clf', RandomForestClassifier(n_estimators=200, random_state=42))])
        pipe_ds.fit(Xn2_train, y_train)
        ablations['no_ds_idx'] = metrics(y_test, pipe_ds.predict(Xn2_test))
    else:
        ablations['no_ds_idx'] = None

    # 3) remove water + ds_idx
    cols_remove = water_cols + (['ds_idx'] if 'ds_idx' in X.columns else [])
    Xn3_train = X_train.drop(columns=cols_remove)
    Xn3_test = X_test.drop(columns=cols_remove)
    numeric_cols = [c for c in Xn3_train.columns if Xn3_train[c].dtype.kind in 'fi']
    categorical_cols = [c for c in Xn3_train.columns if c not in numeric_cols]
    preproc3 = ColumnTransformer([('num', numeric_pipeline, numeric_cols), ('cat', categorical_pipeline, categorical_cols)])
    pipe_nwds = Pipeline([('preproc', preproc3), ('clf', RandomForestClassifier(n_estimators=200, random_state=42))])
    pipe_nwds.fit(Xn3_train, y_train)
    ablations['no_water_no_ds_idx'] = metrics(y_test, pipe_nwds.predict(Xn3_test))

    out = {
        'full_model_metrics': full_metrics,
        'no_water_model_metrics': no_water_metrics,
        'ablations': ablations,
        'water_columns': water_cols,
        'permutation_importance': sorted(importances.items(), key=lambda x: -x[1])[:20]
    }

    Path(ROOT / 'artifacts' / 'leakage_analysis.json').write_text(json.dumps(out, indent=2))
    print('Wrote artifacts/leakage_analysis.json')


if __name__ == '__main__':
    main()
