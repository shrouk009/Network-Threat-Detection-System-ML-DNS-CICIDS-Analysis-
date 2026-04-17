from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_PATH = Path("models/model.pkl")


def model_exists() -> bool:
    return MODEL_PATH.exists()


def _eval_metrics(y_true, y_pred, class_names: Optional[List[str]]) -> Dict:
    merged = pd.concat([pd.Series(y_true).astype(int), pd.Series(y_pred).astype(int)], ignore_index=True)
    labels = sorted(merged.unique())
    if class_names is not None:
        target_names = [class_names[i] if i < len(class_names) else str(i) for i in labels]
    else:
        target_names = [str(i) for i in labels]
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": matrix,
        "classification_report": report,
    }


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    class_names: Optional[List[str]] = None,
    X_eval: Optional[pd.DataFrame] = None,
    y_eval: Optional[pd.Series] = None,
) -> Tuple[Dict, Dict]:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if X_eval is None or y_eval is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        split_mode = "random_split"
    else:
        X_train, y_train = X, y
        X_test, y_test = X_eval, y_eval
        split_mode = "file_based_split"

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1)),
        ]
    )
    pipeline.fit(X_train, y_train)
    test_preds = pipeline.predict(X_test)

    eval_results = _eval_metrics(y_test, test_preds, class_names)
    metrics = {
        "split_mode": split_mode,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "accuracy": eval_results["accuracy"],
        "confusion_matrix": eval_results["confusion_matrix"],
        "classification_report": eval_results["classification_report"],
    }
    bundle = {"pipeline": pipeline, "class_names": class_names}
    joblib.dump(bundle, MODEL_PATH)
    return bundle, metrics