"""Quick end-to-end sanity checks (no Streamlit needed): python3 test_pipeline.py"""
import json

import joblib
import pandas as pd

from features import FEATURE_NAMES, InvalidPolicy, extract_features, parse_policy
from rules import explain

bundle = joblib.load("model.pkl")
model = bundle["model"]


def predict(pol):
    f = extract_features(pol)
    row = pd.DataFrame([[f[n] for n in FEATURE_NAMES]], columns=FEATURE_NAMES)
    p = model.predict_proba(row)[0]
    return model.classes_[p.argmax()], p.max(), f


# The worked example from the project doc must give exactly the doc's feature vector
doc_example = {"Statement": [{"Effect": "Allow",
                              "Action": ["iam:CreateRole", "iam:AttachRolePolicy", "s3:*"],
                              "Resource": "*"}]}
f = extract_features(doc_example)
assert {k: f[k] for k in list(f)[:8]} == {"statement_count": 1, "allow_count": 1, "wildcard_action": 1, "wildcard_resource": 1,
             "service_count": 2, "iam_action_count": 2, "sts_action_count": 0, "has_condition": 0}, f
print("feature extraction matches the doc's example: OK")

cases = {
    "narrow s3 read": ({"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Action": "s3:GetObject",
                        "Resource": "arn:aws:s3:::company-documents/*"}]}, "Low"),
    "admin *:*": ({"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Action": "*", "Resource": "*"}]}, "High"),
    "doc example": (doc_example, "High"),
}
for name, (pol, expected) in cases.items():
    label, p, _ = predict(pol)
    status = "OK" if label == expected else f"MISMATCH (expected {expected})"
    print(f"{name:<16} -> {label} ({p:.0%})  {status}")

# invalid inputs must be rejected, not crash
for bad in ["{not json", "[]", '{"Version": "2012-10-17"}', json.dumps({"Statement": [{"Effect": "Maybe"}]})]:
    try:
        parse_policy(bad)
        raise SystemExit(f"should have rejected: {bad}")
    except InvalidPolicy:
        pass
print("invalid JSON / malformed policies rejected: OK")

factors, recs = explain(doc_example)
print("\nExplanation for the doc example:")
for x in factors:
    print("  -", x)
for x in recs:
    print("  ->", x)
