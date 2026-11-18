from src.preprocessing import load_and_clean, build_feature_matrix
from src.train_model import pick_dataset

df, _ = load_and_clean(pick_dataset())
X, y = build_feature_matrix(df)
X["label"] = y.values  # 0 = legitimate, 1 = phishing

print("Average feature values per class:")
print(X.groupby("label").mean().T.round(2))
print("\nShare of URLs with path_length == 0:")
print(X.groupby("label")["path_length"].apply(lambda s: (s == 0).mean()).round(3))
print("\nSample legitimate URLs:")
print(df[df.is_phishing == 0].url.sample(8, random_state=1).to_string(index=False))
print("\nSample phishing URLs:")
print(df[df.is_phishing == 1].url.sample(8, random_state=1).to_string(index=False))