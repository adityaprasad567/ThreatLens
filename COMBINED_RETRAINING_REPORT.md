# ThreatLens combined-data retraining report

## Sources
- PhiUSIIL_original.csv: 235,795 rows
- PhiLegit-100K.csv: 100,000 rows
- Source label convention: 1=legitimate, 0=phishing. Training convention: 1=phishing, 0=legitimate.
- Combined input: 254,987 unique URLs, after removing duplicate URLs across sources; no conflicting URL labels were found.
- Original PhiLegit-100K source_dataset and features_json metadata are not used as model inputs to prevent source-based shortcuts and ensure inference compatibility.

## Split and training
- Train: 204,007; test: 50,980; overlap of approximate registrable domains: 0
- Group split seed: 81
- Model: Hashed Character N-gram + URL lexical/domain features (SGD logistic)
- URL n-grams, lexical, domain, path and query features

## Held-out results
- Accuracy: 0.903805
- Precision: 0.968450
- Recall: 0.820865
- F1: 0.888571
- ROC-AUC: 0.949062
- False negative rate: 0.179135

## Reproduce
From the ThreatLens folder:
```bash
pip install -r requirements.txt
python -m src.train_model --data data/Combined_PhiUSIIL_PhiLegit.csv
python -m pytest -q tests/test_retrained_model.py
streamlit run app.py
```

## Caveats
The registrable-domain extractor uses a partial suffix list, not the complete Public Suffix List. The domain-grouped holdout is not a time-separated external validation. Dataset labels were not independently reverified. Dashboard source files are unchanged; full Streamlit UI integration was not exercised here.
