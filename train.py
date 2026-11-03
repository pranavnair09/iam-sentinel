"""Train + compare Logistic Regression (baseline) vs Random Forest (main model).

Splitting is GROUPED: all variations of one base policy stay in the same split,
so the test set contains only base policies the model has never seen.
"""
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_NAMES

SEED = 42
LABELS = ["Low", "Medium", "High"]

df = pd.read_csv("policies.csv")
X, y, groups = df[FEATURE_NAMES], df["label"], df["group"]

# ---- grouped 80/20 split ---------------------------------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
tr, te = next(gss.split(X, y, groups))
X_tr, X_te, y_tr, y_te = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
assert set(groups.iloc[tr]).isdisjoint(set(groups.iloc[te])), "group leakage!"
print(f"Train: {len(tr)} policies / {groups.iloc[tr].nunique()} bases   "
      f"Test: {len(te)} policies / {groups.iloc[te].nunique()} bases (no overlap)\n")

models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, random_state=SEED, class_weight="balanced"),
}

results, fitted = {}, {}
cv = GroupKFold(n_splits=5)
for name, model in models.items():
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    cv_f1 = cross_val_score(model, X_tr, y_tr, groups=groups.iloc[tr], cv=cv, scoring="f1_macro")
    results[name] = {
        "accuracy": accuracy_score(y_te, pred),
        "f1_macro": f1_score(y_te, pred, average="macro"),
        "cv_f1_macro_mean": cv_f1.mean(),
        "cv_f1_macro_std": cv_f1.std(),
    }
    fitted[name] = model
    print(f"=== {name} ===")
    print(classification_report(y_te, pred, labels=LABELS, digits=3, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred; order Low/Medium/High):")
    print(pd.DataFrame(confusion_matrix(y_te, pred, labels=LABELS), index=LABELS, columns=LABELS), "\n")

summary = pd.DataFrame(results).T.round(3)
print("=== Comparison (held-out test set) ===")
print(summary.to_string(), "\n")

# Selection rule (fixed here, stated in the report): pick the higher macro-F1, but if the
# two are within 0.01 (a tie on this small test set) keep Random Forest, the project's main
# model, because it captures feature interactions and reports feature importances.
best_name = max(results, key=lambda n: results[n]["f1_macro"])
gap = abs(results["Random Forest"]["f1_macro"] - results["Logistic Regression"]["f1_macro"])
if gap < 0.01:
    best_name = "Random Forest"
    print(f"Models tie (macro-F1 gap {gap:.3f} < 0.01): keeping Random Forest as the main model")
print(f"Selected: {best_name}")
best = fitted[best_name]
if best_name == "Random Forest":
    fi = sorted(zip(FEATURE_NAMES, best.feature_importances_), key=lambda t: -t[1])
    print("\nRandom Forest feature importances:")
    for n, v in fi:
        print(f"  {n:<18} {v:.3f}")

joblib.dump({"model": best, "features": FEATURE_NAMES, "classes": list(best.classes_),
             "name": best_name}, "model.pkl")
with open("metrics.json", "w") as f:
    json.dump({"selected": best_name, "results": results,
               "n_train": len(tr), "n_test": len(te)}, f, indent=2)
print("\nSaved model.pkl and metrics.json")
