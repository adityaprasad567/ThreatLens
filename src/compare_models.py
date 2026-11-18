"""Compare Logistic Regression and Random Forest on URL lexical features.

Run: python -m src.compare_models --data data/Combined_PhiUSIIL_PhiLegit.csv
The deployed classifier remains unchanged; comparison results are saved separately.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import MODELS_DIR
from .feature_extraction import MODEL_FEATURE_NAMES, FEATURE_NAMES, extract_feature_frame, normalize_scheme
from .preprocessing import load_and_clean
from .train_model import registrable_domain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--max-rows', type=int, default=30000)
    args = parser.parse_args()
    if args.max_rows < 100:
        parser.error('--max-rows must be at least 100')

    df, _ = load_and_clean(Path(args.data))
    # Deterministic balanced sample to make training feasible on a laptop.
    per_class = args.max_rows // 2
    df = pd.concat([
        group.sample(n=min(per_class, len(group)), random_state=42)
        for _, group in df.groupby('is_phishing')
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Extract all 12 features; HTTPS is deliberately excluded from model inputs
    # to avoid a misleading dataset shortcut. The UI can still display it.
    X = extract_feature_frame(df['url'].map(normalize_scheme))
    X = X[MODEL_FEATURE_NAMES]
    y = df['is_phishing']
    groups = df['url'].map(registrable_domain)
    train_idx, test_idx = next(GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=42
    ).split(X, y, groups))
    x_train, x_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError('Both classes are required in training and testing. Try a larger --max-rows.')

    classifiers = {
        'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42
        )),
        'Random Forest (Bagging Ensemble)': RandomForestClassifier(
            n_estimators=100, max_depth=18, min_samples_leaf=2,
            class_weight='balanced_subsample', random_state=42, n_jobs=-1
        ),
    }
    scores = {}
    for name, classifier in classifiers.items():
        print(f'Training {name} ...', flush=True)
        classifier.fit(x_train, y_train)
        predicted = classifier.predict(x_test)
        scores[name] = {
            'accuracy': float(accuracy_score(y_test, predicted)),
            'precision': float(precision_score(y_test, predicted, zero_division=0)),
            'recall': float(recall_score(y_test, predicted, zero_division=0)),
            'f1': float(f1_score(y_test, predicted, zero_division=0)),
        }
    result = {
        'trained_at': datetime.now(timezone.utc).isoformat(),
        'dataset': Path(args.data).name,
        'sample_rows': len(df), 'train_rows': len(train_idx), 'test_rows': len(test_idx),
        'split': 'domain-grouped holdout',
        'extracted_features': FEATURE_NAMES,
        'trained_numeric_features': MODEL_FEATURE_NAMES,
        'excluded_from_training': 'uses_https (computed, but excluded to avoid shortcut learning)',
        'models': scores,
        'note': 'Comparison models are separate from the deployed SGD character n-gram classifier.',
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / 'comparison_metrics.json'
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    print(f'Saved {path}')

if __name__ == '__main__':
    main()
