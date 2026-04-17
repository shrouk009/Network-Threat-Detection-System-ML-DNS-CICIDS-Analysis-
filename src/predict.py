from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np

MODEL_PATH = Path("models/model.pkl")


def load_model() -> Dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file not found at models/model.pkl. Train first.")
    return joblib.load(MODEL_PATH)


def predict(model_bundle: Dict, X) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    model = model_bundle["pipeline"]
    preds = model.predict(X)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
    else:
        class_count = len(np.unique(preds))
        probs = np.zeros((len(preds), max(class_count, 2)))
        for idx, pred in enumerate(preds):
            probs[idx, int(pred)] = 1.0

    class_names = model_bundle.get("class_names")
    if class_names is None:
        class_names = [str(c) for c in sorted(np.unique(preds))]
    predicted_labels = [class_names[int(i)] if int(i) < len(class_names) else str(i) for i in preds]
    return preds, probs, predicted_labels