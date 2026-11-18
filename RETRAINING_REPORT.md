# ThreatLens retraining report

## Reproduce

Install dependencies from `requirements.txt`, then from this folder run:

```bash
python -m src.train_model --data data/PhiUSIIL_original.csv
```

The command writes `models/best_model.joblib` and `models/metrics.json`. The metrics file records the SHA-256 of the source CSV, cleaning counts, split seed, group overlap, confusion matrix and classification metrics.

## Training design

- Source: supplied PhiUSIIL CSV (235,795 rows).
- Cleaning: removed 425 exact duplicate URL records; no rows with missing URL/label and no conflicting exact URL labels remained after cleaning.
- Label mapping: PhiUSIIL label 0 = phishing, label 1 = legitimate; converted to ThreatLens convention 1 = phishing, 0 = legitimate.
- Split: deterministic GroupShuffleSplit, grouped by approximate registrable domain; zero domain overlap between train and test.
- Features: hashed character 3–5-grams over the complete normalized URL (hostname, path and query), plus shared numeric URL lexical/domain features.
- Classifier: SGDClassifier with logistic loss and balanced class weights.
- No synthetic popular-domain augmentation was used.

## Held-out domain-grouped results

| Metric | Result |
|---|---:|
| Accuracy | 89.95% |
| Precision (phishing) | 94.51% |
| Recall (phishing) | 81.06% |
| F1 (phishing) | 87.27% |
| ROC-AUC | 94.17% |
| False-negative rate | 18.94% |

Confusion matrix (actual rows / predicted columns, labels legitimate then phishing):

- TN: 26,007
- FP: 938
- FN: 3,769
- TP: 16,135

The test set has 46,849 rows and 33,921 registrable domains; train/test domain overlap is zero.

## Important caveats

- Registrable-domain extraction uses a small built-in two-part suffix list, not the complete Public Suffix List. Domain grouping is therefore an approximation.
- The supplied dataset is a static benchmark. These metrics do not establish performance on current live phishing campaigns or future domains.
- PhiUSIIL contains dataset-specific URL patterns. A separate time-based and independently sourced external test is recommended before production use.
- The model uses URL text only; it does not visit websites, inspect page content, validate certificates, or query threat-intelligence feeds.
- Model serialization was tested in the build environment with scikit-learn 1.8.0. The project requirements pin 1.9.1; retrain in the target environment if joblib reports a scikit-learn version warning.
