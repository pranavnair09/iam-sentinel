"""IAM Sentinel: Streamlit GUI.   Run with:  streamlit run app.py"""
import json

import joblib
import pandas as pd
import streamlit as st

from features import FEATURE_NAMES, InvalidPolicy, extract_features, parse_policy
from rules import explain

st.set_page_config(page_title="IAM Sentinel", page_icon="🛡️", layout="centered")

COLORS = {"Low": "#2e7d32", "Medium": "#ef6c00", "High": "#c62828"}

EXAMPLES = {
    "Low: read one bucket": {
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow", "Action": "s3:GetObject",
                       "Resource": "arn:aws:s3:::company-documents/*"}]},
    "Medium: read/write/delete one bucket": {
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow",
                       "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
                       "Resource": "arn:aws:s3:::company-documents/*"}]},
    "High: IAM + S3 wildcards on everything": {
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow",
                       "Action": ["iam:CreateRole", "iam:AttachRolePolicy", "s3:*"],
                       "Resource": "*"}]},
    "High: full admin": {
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow", "Action": "*", "Resource": "*"}]},
}


@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


bundle = load_model()
model = bundle["model"]

st.title("🛡️ IAM Sentinel")
st.caption("ML-powered AWS IAM policy risk analyzer. Paste a policy; nothing is sent to AWS.")

choice = st.selectbox("Load an example (optional)", ["(none)"] + list(EXAMPLES))
default = json.dumps(EXAMPLES[choice], indent=2) if choice != "(none)" else ""
text = st.text_area("Paste IAM policy JSON", value=default, height=260, key=f"policy_{choice}")

if st.button("Analyze Policy", type="primary"):
    try:
        policy = parse_policy(text)
    except InvalidPolicy as e:
        st.error(f"**Invalid IAM policy.** {e}")
        st.stop()

    feats = extract_features(policy)
    row = pd.DataFrame([[feats[n] for n in FEATURE_NAMES]], columns=FEATURE_NAMES)
    probs = model.predict_proba(row)[0]
    classes = list(model.classes_)
    label = classes[probs.argmax()]
    p = probs.max()

    st.markdown(
        f"""<div style="border:2px solid {COLORS[label]};border-radius:12px;padding:16px 20px;margin:12px 0;">
        <div style="font-size:0.85rem;opacity:.7">Predicted risk level</div>
        <div style="font-size:2.2rem;font-weight:700;color:{COLORS[label]}">{label.upper()}</div>
        <div>Predicted probability: <b>{p:.0%}</b>
        <span style="opacity:.6">(model estimate, not certainty)</span></div></div>""",
        unsafe_allow_html=True)

    st.progress(0.0, text="Class probabilities")
    st.bar_chart(pd.Series({c: float(v) for c, v in zip(classes, probs)}).reindex(["Low", "Medium", "High"]))

    factors, recs = explain(policy)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Risk factors")
        for f in factors:
            st.markdown(f"- {f}")
    with c2:
        st.subheader("Recommendations")
        for r in recs:
            st.markdown(f"- {r}")

    with st.expander("Features the model saw"):
        st.dataframe(row.T.rename(columns={0: "value"}))
    st.caption(f"Classifier: {bundle['name']}. The explanation panel is rule-based and separate from the ML prediction.")
