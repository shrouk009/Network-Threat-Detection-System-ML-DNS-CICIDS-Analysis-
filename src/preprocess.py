from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

LABEL_CANDIDATES = ["label", "class", "target"]


def _find_label_column(df: pd.DataFrame) -> Optional[str]:
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for candidate in LABEL_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]
    return None


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series], Dict]:
    data = df.copy()
    data.columns = [str(c).strip() for c in data.columns]

    label_col = _find_label_column(data)
    y_encoded = None
    y_raw = None
    label_encoder = None

    if label_col is not None:
        y_raw = data[label_col].astype(str).str.strip()
        label_encoder = LabelEncoder()
        y_encoded = pd.Series(label_encoder.fit_transform(y_raw), name="encoded_label")
        data = data.drop(columns=[label_col])

    X = data.select_dtypes(include=["number"]).copy()
    if X.empty:
        raise ValueError("No numeric features found after preprocessing.")

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)

    metadata = {
        "rows": len(X),
        "features": X.shape[1],
        "label_column": label_col,
        "label_encoder": label_encoder,
        "y_raw": y_raw,
        "feature_names": list(X.columns),
    }
    return X, y_encoded, metadata