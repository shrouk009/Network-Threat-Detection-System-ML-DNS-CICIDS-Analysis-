from pathlib import Path
from typing import List, Literal, Tuple

import pandas as pd


def list_data_files(data_dir: str = "data") -> List[str]:
    base = Path(data_dir)
    if not base.exists():
        return []
    return sorted(str(p) for p in base.glob("*.csv"))


def load_data(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    if path.suffix.lower() != ".csv":
        raise ValueError("Only CSV files are supported in this workflow.")
    return pd.read_csv(path)


def load_all_data(data_dir: str = "data") -> Tuple[pd.DataFrame, List[str]]:
    files = list_data_files(data_dir)
    if not files:
        raise FileNotFoundError("No CSV files found in data directory.")
    frames = [load_data(path) for path in files]
    return pd.concat(frames, ignore_index=True), files


def detect_file_type(file_path: str, dataframe: pd.DataFrame) -> Literal["Monday", "Friday_DDoS", "Unknown"]:
    file_name = Path(file_path).name.lower()
    columns_text = " ".join(str(c).lower() for c in dataframe.columns)
    if "monday" in file_name:
        return "Monday"
    if "friday" in file_name and "ddos" in file_name:
        return "Friday_DDoS"
    if "ddos" in columns_text:
        return "Friday_DDoS"
    return "Unknown"