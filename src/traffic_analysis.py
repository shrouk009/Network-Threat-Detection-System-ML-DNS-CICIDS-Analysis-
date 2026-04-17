from typing import Dict, List

import pandas as pd


def _safe_col(df: pd.DataFrame, key: str):
    mapping = {str(c).strip().lower(): c for c in df.columns}
    return mapping.get(key.lower())


def analyze_traffic_patterns(df: pd.DataFrame, predicted_labels: List[str]) -> Dict:
    result = {}
    total = len(df)
    result["total_flows"] = int(total)

    fd_col = _safe_col(df, "flow duration")
    if fd_col is not None:
        result["avg_flow_duration"] = float(pd.to_numeric(df[fd_col], errors="coerce").fillna(0).mean())
    else:
        result["avg_flow_duration"] = 0.0

    port_col = _safe_col(df, "destination port")
    if port_col is not None:
        top_ports = pd.to_numeric(df[port_col], errors="coerce").dropna().astype(int).value_counts().head(10)
        result["top_destination_ports"] = {int(k): int(v) for k, v in top_ports.items()}
    else:
        result["top_destination_ports"] = {}

    pred_series = pd.Series(predicted_labels)
    result["predicted_class_distribution"] = {
        str(k): int(v) for k, v in pred_series.value_counts().to_dict().items()
    }
    return result
