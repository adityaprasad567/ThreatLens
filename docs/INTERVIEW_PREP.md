# ThreatLens: Interview Preparation Guide

Use this to explain the project honestly and confidently. Replace every `[your result]` with numbers from **your own training run** (Model Performance page or `models/metrics.json`). Never quote numbers you have not produced.

---

## 1. The project in simple language

ThreatLens checks whether a web link looks like a phishing link. You paste a URL. The app measures 12 things about the text of the link (how long it is, how many hyphens, whether it uses an IP address, and so on), feeds them to a trained machine-learning model, and shows a verdict, a confidence score, a risk level (Low, Medium, High) and a list of the specific things that look suspicious. It never opens the website, so it is safe to use on dangerous links.

## 2. How it works internally

1. **Data**: a public phishing URL dataset (PhiUSIIL, about 235,000 labelled URLs).
2. **Preprocessing** (`preprocessing.py`): remove missing values, exact duplicates and URLs that appear with conflicting labels, then convert labels to one convention (1 = phishing).
3. **Feature extraction** (`feature_extraction.py`): turn each URL into 12 numbers.
4. **Training** (`train_model.py`): stratified 80/20 split, train Logistic Regression and Random Forest, evaluate both on the unseen 20%.
5. **Model selection**: pick the model with the best F1-score, save it with Joblib along with `metrics.json`.
6. **Prediction** (`prediction.py`): validate the URL, extract the same URL features (plus the domain text for the n-gram model), get the phishing probability, derive prediction, confidence and risk level.
7. **Explanation** (`threat_analysis.py`): rule-based indicators (for example "contains @", "IP address host") explain the result.
8. **Storage** (`database.py`): each scan is saved to SQLite and shown in the dashboard.

**Key design point:** the same `extract_features()` function is used in training and prediction. If the two differed, the model would receive inputs it was never trained on (training-serving skew).

## 3. How feature extraction works

| Feature | Why it can matter |
|---|---|
| URL length | Phishing URLs are often long to hide the real destination |
| Domain length | Look-alike or auto-generated domains tend to be long |
| Number of dots | Many dots usually means many subdomain levels |
| Number of hyphens | `secure-bank-login-update` style domains |
| Number of subdomains | `bank.com.attacker.xyz` hides the real domain (`www` is ignored) |
| HTTPS | Absence is a weak signal (many phishing sites now use HTTPS too) |
| IP address host | Legitimate services rarely use raw IPs; also detects decimal/hex forms |
| `@` symbol | Browsers ignore text before `@`, so it can disguise the destination |
| Special characters | Encoded or obfuscated parameters |
| Suspicious keywords | login, verify, secure, account, and similar lures (22 words) |
| Path length | Long, random-looking paths |
| Number of digits | Auto-generated IDs and random strings |

Implementation details worth knowing: URLs are parsed with `urllib.parse`; subdomains are counted by splitting the host and handling a short list of two-part suffixes such as `co.uk`; "special characters" means anything that is not a letter, digit, or one of `. - / :`.

## 4. Why Logistic Regression and Random Forest

- **Logistic Regression** is a simple linear baseline. It is fast, needs scaled inputs (I use `StandardScaler`), and its coefficients are interpretable. It shows how far a linear boundary gets you.
- **Random Forest** is an ensemble of decision trees. It captures non-linear patterns and interactions (for example "long URL AND no HTTPS AND keywords"), handles features on different scales without scaling, and is robust to noise. It also gives feature importance.
- Comparing a simple and a stronger model is good practice: it tells you whether the extra complexity is justified.
- Both use `class_weight="balanced"` so the smaller class is not ignored.

## 5. How performance is evaluated

Phishing is the positive class.

- **Accuracy**: share of all predictions that were correct. Can mislead if classes are imbalanced.
- **Precision**: of the URLs flagged as phishing, how many really were.
- **Recall**: of all real phishing URLs, how many were caught.
- **F1-score**: harmonic mean of precision and recall. I select the best model by F1 (ties broken by recall).
- **Confusion matrix**: the four counts TN, FP, FN, TP.
- **ROC-AUC**: how well the model ranks phishing above legitimate across all thresholds.

All are computed on a **held-out test set** the model never saw during training. Results are written to `metrics.json` by the training script and read by the dashboard, so nothing is typed in by hand.

My results: Character N-gram model, accuracy 0.883, precision 0.821, recall 0.761, F1 0.790, ROC-AUC 0.913 on the PhiUSIIL test set (domain-only scoring, http/https ignored). Re-run `python -m src.train_model` and check `models/metrics.json` before quoting.

## 6. What is phishing

Phishing is a social-engineering attack where the attacker impersonates a trusted organization (bank, delivery service, employer, government) to trick a victim into revealing credentials, card details or OTPs, or into installing malware. It usually arrives by email, SMS, messaging apps or QR codes and points to a fake website. Variants: spear phishing (targeted), whaling (executives), smishing (SMS), vishing (voice).

## 7. False positives and false negatives

- **False positive**: a legitimate URL flagged as phishing. Cost: blocked legitimate site, annoyed users, "alert fatigue".
- **False negative**: a phishing URL judged legitimate. Cost: a user may enter credentials on a fake site. In security this is usually the more dangerous error.
- They trade off against each other through the decision threshold. Lowering the threshold raises recall and lowers precision.
- In ThreatLens, a "legitimate" verdict that coexists with a serious red flag (IP host or `@`) is shown as "check this link" rather than a green verdict. That is a deliberate guard against false negatives.

## 8. Limitations of URL-based detection

- Sees only text: no page content, certificate, domain age, WHOIS, redirects or reputation.
- Attackers can compromise a legitimate site, which gives a clean-looking URL.
- URL shorteners hide the real destination.
- Dataset bias: benchmark URLs come from specific sources, so accuracy may overstate real-world accuracy.
- Concept drift: phishing patterns change, so models need retraining.
- Keywords are language-specific and also appear on genuine pages.
- Output is a probability, not verified threat intelligence.

---

## 9. Twenty-five technical questions

**1. What problem does ThreatLens solve?**
It classifies a URL as phishing or legitimate using only the URL text, gives a confidence and risk level, and explains which features looked suspicious. It is a lightweight first-pass signal, not a replacement for blocklists.

**2. Why not just use a blocklist?**
Blocklists only know URLs already reported, so new phishing URLs get through. A model can generalize to unseen URLs. In production you would combine both.

**3. Which dataset did you use and why?**
PhiUSIIL from the UCI repository: about 134,850 legitimate and 100,945 phishing URLs, recent, large, CC BY 4.0 licensed, and peer-reviewed (Prasad & Chandra, 2024).

**4. How did you clean the data?**
Dropped missing and empty rows, removed exact duplicates, removed URLs that appeared with both labels (unreliable), and mapped labels to one convention. PhiUSIIL uses 1 = legitimate, which I flip to 1 = phishing. A cleaning report is stored in `metrics.json`.

**5. Why did duplicates matter?**
If the same URL lands in both training and test sets, the test score is inflated because the model has memorized it. This is data leakage.

**6. How do you avoid training-serving skew?**
One shared `extract_features()` function used by both the training script and the prediction module, with the column order defined once in `FEATURE_NAMES`.

**7. Explain your 12 features and which matter most.**
See section 3. Which matter most depends on the data. The Model Performance page shows Random Forest importance computed from the trained model; I would cite that page rather than guess.

**8. Why is HTTPS only a weak signal?**
Free certificates make HTTPS easy for attackers. Its absence is mildly suspicious, but its presence proves nothing. The app says this explicitly.

**9. How do you detect an IP address host?**
`ipaddress.ip_address()` for IPv4/IPv6, plus checks for all-digit hosts (decimal-encoded IPs) and `0x...` hex forms that attackers use to obfuscate.

**10. How do you count subdomains without an external library?**
Split the host on dots, treat the last two labels as the registered domain (three for known suffixes such as `co.uk`), and count the rest, ignoring a leading `www`. The suffix list is short and not exhaustive, which is a known limitation; `tldextract` would fix it.

**11. Why Logistic Regression?**
A fast, interpretable baseline. It shows how well a linear boundary separates the classes and sets a bar the complex model must beat.

**12. Why Random Forest?**
It models non-linear interactions, needs no feature scaling, resists overfitting through averaging many trees, and provides feature importance.

**13. Why scale features for Logistic Regression but not Random Forest?**
Logistic Regression's optimization and regularization are sensitive to feature scale. Trees split on thresholds, so monotonic scaling does not change them.

**14. What is `class_weight="balanced"`?**
It weights errors on the rarer class more heavily so the model does not simply favor the majority class.

**15. Why stratified splitting?**
It keeps the phishing/legitimate ratio the same in training and test sets, so the evaluation is representative.

**16. Why F1 to choose the model?**
Accuracy hides class imbalance, and security cares about both missing phishing (recall) and false alarms (precision). F1 balances them. If missing phishing is costlier, I would weight recall more.

**17. Explain the confusion matrix for your best model.**
Describe your own numbers: `[your result: TN, FP, FN, TP]`. Rows are actual, columns are predicted. FP are legitimate URLs wrongly flagged; FN are phishing URLs missed.

**18. How is "confidence" defined?**
The probability of the predicted class: `max(p, 1 - p)`. It is a model probability, not a guarantee. Tree ensembles are not always well calibrated, which is why calibration is listed as future work.

**19. How is risk level decided?**
From the phishing probability: below 0.35 Low, 0.35 to 0.65 Medium, 0.65 and above High. The thresholds are in `config.py` and are a product choice, not a learned value.

**20. Are your "indicators" the model's reasoning?**
No. The verdict comes from the model. Indicators are simple rule-of-thumb checks used to explain the result in human terms. The page also compares the URL with per-class training averages. For true per-prediction attribution I would add SHAP.

**21. What if the model says "legitimate" but the URL has an `@` symbol?**
The UI shows "Model says legitimate, but check this link" instead of a green verdict, because a legitimate verdict next to a serious red flag should not reassure the user. The probability itself is not altered.

**22. How do you validate input safely?**
Scheme must be http/https, no whitespace, a maximum length, a parseable host and port, and the hostname must be a valid domain (IDNA-encoded) or IP. Invalid input returns a clear message and is never scored or saved. The URL is never fetched, so there is no SSRF or malware-download risk.

**23. Why SQLite and what is stored?**
It is built into Python, needs no server, and suits a single-user local app. Stored: URL, prediction, risk level, confidence, phishing probability, timestamp. Parameterized queries are used to prevent SQL injection. History is local only and can be cleared.

**24. How would you deploy or scale this?**
Wrap `predict_url` in an API (for example FastAPI), containerize it, load the model once at start-up, and add caching and rate limiting. Retrain on a schedule with fresh data and monitor drift. These were deliberately out of scope here.

**25. How would you improve it?**
Character n-gram features, gradient boosting, probability calibration, time-based validation to measure drift, WHOIS/DNS and reputation enrichment kept separate from the ML score, SHAP explanations, and testing on a second independent dataset to check generalization.

---

## 10. Your two-minute Deloitte explanation

> "ThreatLens is a phishing URL detection system I built to understand how machine learning can support cyber defense.
>
> The problem is that phishing is still one of the most common attack vectors, and blocklists only catch links that have already been reported. I wanted to see how well a model can judge a brand-new link just from how it looks, and, importantly, explain its decision.
>
> I trained on the public PhiUSIIL dataset from UCI, about 235,000 labelled URLs. I cleaned it by removing missing values, duplicates and conflicting labels, then extracted 12 lexical features such as URL length, number of hyphens and subdomains, whether it uses an IP address or an at-symbol, and suspicious keywords. I trained Logistic Regression and Random Forest as baselines and a character n-gram model on the domain name as the third, compared them on a held-out test set using precision, recall and F1, and saved the best one. The character n-gram model won with F1 0.790 and recall 0.761. I removed the http/https feature after finding the model used it as a shortcut, which is why my score is lower but honest.
>
> The dashboard is built in Streamlit. A user pastes a URL, the app validates it without ever visiting it, predicts phishing or legitimate with a confidence score and risk level, and lists the specific indicators behind the result. Every scan goes into SQLite, and the performance page reads real metrics from the training run.
>
> Two things I am careful about. First, I use the same feature code for training and prediction, to avoid training-serving skew. Second, I am explicit that this is a model prediction, not verified threat intelligence: a URL-only model can't see page content, domain age or reputation, and it can be fooled by a compromised legitimate site. So when the model says 'legitimate' but a serious red flag is present, the app asks the user to check manually rather than showing a green tick.
>
> Next I would add character-level features, calibrate probabilities, and combine the model with reputation feeds, which is closer to how a real security operations workflow would use it."

### Likely follow-ups to prepare

- "What would you change if false negatives were far more costly?" (Lower the threshold, optimize recall, report the precision cost.)
- "How do you know it generalizes?" (Held-out test set; honest about dataset bias; would test on a second dataset and a time-based split.)
- "Show me the code that guarantees consistent features." (`src/feature_extraction.py`, used by `preprocessing.py` and `prediction.py`.)
