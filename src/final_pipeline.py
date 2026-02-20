"""
Early-Stage Lung Cancer Risk Prediction (Research Pipeline)

This file contains a clean, readable experimental pipeline for comparing multiple
classification models under a consistent preprocessing + evaluation setup.

Notes:
- SMOTE is applied inside the pipeline to avoid data leakage during cross-validation.
- Stratified train/test split is used due to class imbalance.
- Metrics include ROC-AUC and PR-AUC (more informative under imbalance), plus Precision/Recall/F1.
- This is a research/educational project; not medical advice.
"""

import warnings
import pandas as pd

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    brier_score_loss,
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# -----------------------------
# 1) Data loading + minimal prep
# -----------------------------
def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Encode target + binary gender (matches original dataset encoding)
    df["LUNG_CANCER"] = df["LUNG_CANCER"].map({"NO": 0, "YES": 1})
    df["GENDER"] = df["GENDER"].map({"F": 0, "M": 1})

    # Keep it simple here; upgrade to imputation if needed
    df = df.dropna().reset_index(drop=True)
    return df


# -----------------------------
# 2) Model definitions (tuned using grid search inside 01_raw_experiment.py)
# -----------------------------
def get_models(random_state: int = 42) -> dict:
    return {
        "Logistic Regression": LogisticRegression(C=1, solver="lbfgs", penalty='l2', random_state=random_state),
        "LDA": LinearDiscriminantAnalysis(solver="lsqr", shrinkage=0.2, tol=0.0001),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=5,
            criterion="entropy",
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            min_samples_split=20,
            max_features="log2",
            random_state=random_state,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric="logloss",
        ),
        "SVM (RBF)": SVC(
            C=1.0,
            gamma="scale",
            kernel="rbf",
            probability=True,
            random_state=random_state,
        ),
    }


# -----------------------------------
# 3) Consistent pipeline construction
# -----------------------------------
def make_pipeline(model, random_state: int = 42):
    """
    Pipeline order:
    - StandardScaler: useful for LR/LDA/SVM (harmless for trees)
    - SMOTE: applied only on training folds inside CV, avoiding leakage
    - Chosen classifier model
    """
    return ImbPipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=random_state)),
            ("clf", model),
        ]
    )


# -----------------------------
# 4) Evaluation helpers
# -----------------------------
def crossval_metrics(pipe, X_train, y_train, cv: int = 5) -> dict:
    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }
    out = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    summary = {}
    for k, v in out.items():
        if k.startswith("test_"):
            m = k.replace("test_", "")
            summary[f"cv_{m}_mean"] = float(v.mean())
            summary[f"cv_{m}_std"] = float(v.std())
    return summary


def test_metrics(fitted_pipe, X_test, y_test) -> dict:
    y_pred = fitted_pipe.predict(X_test)

    metrics = {
        "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }

    if hasattr(fitted_pipe, "predict_proba"):
        y_proba = fitted_pipe.predict_proba(X_test)[:, 1]
        metrics["test_roc_auc"] = float(roc_auc_score(y_test, y_proba))
        metrics["test_pr_auc"] = float(average_precision_score(y_test, y_proba))
        metrics["test_brier"] = float(brier_score_loss(y_test, y_proba))

    return metrics


# -----------------------------
# 5) Main experiment loop
# -----------------------------
def run_experiment(data_path: str = "data/lung_cancer.csv") -> pd.DataFrame:
    df = load_dataset(data_path)
    X = df.drop(columns=["LUNG_CANCER"])
    y = df["LUNG_CANCER"]

    #Hold-out test set for final reporting
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    rows = []
    for model_name, model in get_models(random_state=42).items():
        pipe = make_pipeline(model, random_state=42)

        cv = crossval_metrics(pipe, X_train, y_train, cv=5)

        pipe.fit(X_train, y_train)
        test = test_metrics(pipe, X_test, y_test)

        row = {"model": model_name}
        row.update(cv)

        for k in ["test_roc_auc", "test_pr_auc", "test_brier", "test_precision", "test_recall", "test_f1"]:
            if k in test:
                row[k] = test[k]

        rows.append(row)

        #Optional: keep short console output
        print(f"{model_name}: PR-AUC={row.get('test_pr_auc', float('nan')):.4f}, ROC-AUC={row.get('test_roc_auc', float('nan')):.4f}")

    results = pd.DataFrame(rows)

    #Sort primarily by PR-AUC (useful when positive class is rare)
    sort_cols = [c for c in ["test_pr_auc", "test_roc_auc"] if c in results.columns]
    if sort_cols:
        results = results.sort_values(by=sort_cols, ascending=False)

    return results


if __name__ == "__main__":
    results_df = run_experiment("data/lung_cancer.csv")
    print("\n=== Final Results Table ===")
    print(results_df.to_string(index=False))