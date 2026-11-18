"""Which features carry the score? Retrains a Random Forest on feature subsets.

Run from the project root:  python ablation.py
If F1 stays very high with a single trivial feature (such as uses_https), the
benchmark is easy because of dataset style, not because the model understands phishing.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TEST_SIZE
from src.feature_extraction import FEATURE_NAMES
from src.preprocessing import build_feature_matrix, load_and_clean, prepare_for_training
from src.train_model import pick_dataset

df, report = load_and_clean(pick_dataset())
df, mode = prepare_for_training(df, report)
print(f"Training mode: {mode}, rows: {len(df):,}\n")
X, y = build_feature_matrix(df)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

without = lambda *drop: [f for f in FEATURE_NAMES if f not in drop]
experiments = {
    "All 12 features": FEATURE_NAMES,
    "Without uses_https": without("uses_https"),
    "Only uses_https": ["uses_https"],
    "Only url_length + domain_length": ["url_length", "domain_length"],
    "Without https, hyphens, subdomains, digits": without("uses_https", "num_hyphens", "num_subdomains", "num_digits"),
}

print(f"{'Feature set':<46}{'F1':>8}{'Recall':>9}")
for name, cols in experiments.items():
    rf = RandomForestClassifier(n_estimators=60, max_depth=20, min_samples_leaf=2,
                                class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)
    rf.fit(X_train[cols], y_train)
    pred = rf.predict(X_test[cols])
    print(f"{name:<46}{f1_score(y_test, pred):>8.4f}{recall_score(y_test, pred):>9.4f}")
