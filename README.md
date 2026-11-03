# IAM Sentinel

Python-based supervised ML tool that classifies AWS IAM policies as Low / Medium / High risk
(Logistic Regression baseline vs Random Forest), with a rule-based layer that explains the
risk and gives least-privilege recommendations. Streamlit GUI. No AWS connection needed.

## Run

```bash
pip install scikit-learn pandas joblib streamlit
python3 generate_dataset.py   # builds policies.csv (labeled with the rubric in that file)
python3 train.py              # grouped split, compares both models, saves model.pkl + metrics.json
python3 test_pipeline.py      # sanity checks
streamlit run app.py          # GUI
```

## Files

| File | Role |
| --- | --- |
| `features.py` | Policy JSON -> 12 numeric features (8 core + write/delete/escalation/sensitive counts), plus validation |
| `generate_dataset.py` | Labeling rubric + builds ~900 policies from 130 base policies x 8 variations |
| `policies.csv` | Labeled dataset (includes `group` = base policy id) |
| `train.py` | Grouped train/test split, Logistic Regression vs Random Forest, saves best |
| `rules.py` | Explanations and recommendations (not ML) |
| `app.py` | Streamlit GUI |
| `model.pkl`, `metrics.json` | Trained model and its evaluation numbers |

## Method notes (for the report)

- **Grouped split:** every variation of a base policy stays in the same split (`GroupShuffleSplit`),
  so the test set only has base policies the model never saw. `train.py` asserts there is no overlap.
- **Rubric first:** the Low/Medium/High scoring rubric lives in `generate_dataset.py` and was fixed
  before training. The 4 extra features (write, delete, escalation, sensitive action counts) were added
  after a first run with only the 8 core features reached ~70% accuracy: those 8 cannot tell a
  read-only policy from one that can write/delete, or spot privilege-escalation actions.
  With 12 features: ~93% test accuracy, ~0.97 cross-validated macro-F1.
- **Circularity caveat:** the extra features mirror concepts used in the rubric, which makes the task
  easier. Report this openly; the labels are rubric-based, not ground truth from real incidents.
- **Model comparison:** with 12 features Logistic Regression and Random Forest tie (macro-F1 gap < 0.01).
  `train.py` keeps Random Forest in that case (main model; handles interactions; gives feature importances).
  Report it as: a simple linear model is already sufficient once domain-aware features are used.
- **Synthetic data:** policies are generated, not collected from real AWS accounts. State this as a
  limitation. Results show the approach works on this distribution, not that it generalises to all real policies.
- **Small test set:** ~26 base policies in the test split, so single-split numbers are noisy. The
  GroupKFold cross-validation numbers in `metrics.json` are a more stable estimate.
