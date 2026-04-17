from pathlib import Path

import pandas as pd

from src.loader import detect_file_type, list_data_files, load_all_data, load_data
from src.predict import load_model, predict
from src.preprocess import preprocess_data
from src.reporting import OutputManager
from src.rules import simulate_detection_rules
from src.traffic_analysis import analyze_traffic_patterns
from src.train import model_exists, train_model


def _run_file_based_evaluation(train_file: str, eval_file: str, output: OutputManager):
    print(f"[INFO] File-based evaluation: train on {train_file}, test on {eval_file}")
    train_df = load_data(train_file)
    eval_df = load_data(eval_file)

    X_train, _, train_meta = preprocess_data(train_df)
    X_eval, _, eval_meta = preprocess_data(eval_df)
    y_train_raw = train_meta.get("y_raw")
    y_eval_raw = eval_meta.get("y_raw")

    if y_train_raw is None or y_eval_raw is None:
        raise ValueError("Both train and eval files must contain label column.")

    class_names = sorted(pd.Series(y_train_raw).astype(str).str.strip().unique().tolist())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    train_series = pd.Series(y_train_raw).astype(str).str.strip()
    train_mask = train_series.isin(class_to_idx)
    X_train = X_train[train_mask.values].reset_index(drop=True)
    y_train = train_series[train_mask].map(class_to_idx).astype(int).reset_index(drop=True)

    eval_series = pd.Series(y_eval_raw).astype(str).str.strip()
    eval_mask = eval_series.isin(class_to_idx)
    X_eval_known = X_eval[eval_mask.values].reset_index(drop=True)
    y_eval = eval_series[eval_mask].map(class_to_idx).astype(int).reset_index(drop=True)
    unknown_classes = sorted(eval_series[~eval_mask].unique().tolist())
    unknown_rows = int((~eval_mask).sum())
    known_rows = int(eval_mask.sum())
    total_rows = int(len(eval_series))

    model_bundle, train_metrics = train_model(
        X_train, y_train, class_names=class_names, X_eval=X_eval_known, y_eval=y_eval
    )
    train_metrics["eval_total_rows"] = total_rows
    train_metrics["eval_known_rows"] = known_rows
    train_metrics["eval_unknown_rows"] = unknown_rows
    train_metrics["eval_unknown_classes"] = unknown_classes

    # Predict over ALL eval rows so report reflects true traffic volume.
    _, probs, predicted_labels = predict(model_bundle, X_eval)
    traffic_summary = analyze_traffic_patterns(eval_df, predicted_labels)
    rules_summary = simulate_detection_rules(eval_df)

    eval_name = Path(eval_file).stem.replace(".pcap_ISCX", "").replace("-", "_")
    chart_path = output.save_visualization(predicted_labels, filename=f"attack_results_{eval_name}.png")
    report_path = output.save_report(
        predicted_labels=predicted_labels,
        probabilities=probs,
        true_labels=eval_series.reset_index(drop=True),
        traffic_analysis=traffic_summary,
        rule_simulation=rules_summary,
        train_metrics=train_metrics,
        filename=f"security_report_{eval_name}.txt",
    )
    return {"chart_path": chart_path, "report_path": report_path, "eval_name": eval_name, "metrics": train_metrics}


def run_dual_evaluation() -> None:
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("report").mkdir(parents=True, exist_ok=True)
    output = OutputManager("report")

    monday = "data/Monday-WorkingHours.pcap_ISCX.csv"
    friday = "data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    if not Path(monday).exists() or not Path(friday).exists():
        raise FileNotFoundError("Dual evaluation expects Monday and Friday CSV files in data/.")

    print("[INFO] Running dual strict evaluation...")
    res1 = _run_file_based_evaluation(train_file=friday, eval_file=monday, output=output)
    res2 = _run_file_based_evaluation(train_file=monday, eval_file=friday, output=output)

    print("\n====== DUAL EVALUATION OUTPUT ======")
    print(f"Scenario 1 (Friday->Monday) report: {res1['report_path']}")
    print(f"Scenario 1 (Friday->Monday) chart:  {res1['chart_path']}")
    print(f"Scenario 2 (Monday->Friday) report: {res2['report_path']}")
    print(f"Scenario 2 (Monday->Friday) chart:  {res2['chart_path']}")
    print("====================================")


def run_pipeline(file_path: str = "", retrain: bool = False, use_all_data: bool = False) -> None:
    """
    Workflow:
    User selects file -> Loader -> Auto detect -> Preprocess ->
    ML attack detection -> Visualization -> Analyst report.
    """
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("report").mkdir(parents=True, exist_ok=True)

    if use_all_data:
        df, source_files = load_all_data("data")
        file_type = "Mixed_Data"
        print(f"[INFO] Loaded {len(source_files)} files from data/:")
        for p in source_files:
            print(f"  - {p}")
    else:
        df = load_data(file_path)
        file_type = detect_file_type(file_path, df)
        print(f"[INFO] Detected input type: {file_type}")

    X, y, metadata = preprocess_data(df)
    print(f"[INFO] Preprocessed samples: {metadata['rows']}, features: {metadata['features']}")
    output = OutputManager("report")
    inference_df = df
    inference_X = X
    report_true_labels = metadata.get("y_raw")

    if retrain or not model_exists():
        if y is None:
            raise ValueError(
                "No label column found. Cannot train model. "
                "Either provide labeled data or run prediction with an existing model."
            )
        print("[INFO] Training attack classification model...")
        class_names = list(metadata["label_encoder"].classes_) if metadata.get("label_encoder") is not None else None

        X_eval = None
        y_eval = None
        if use_all_data:
            files = list_data_files("data")
            if len(files) >= 2:
                train_file = files[0]
                eval_file = files[-1]
                print(f"[INFO] Strict evaluation mode: train on {train_file}, test on {eval_file}")

                train_df = load_data(train_file)
                eval_df = load_data(eval_file)
                X_train_file, _, train_meta = preprocess_data(train_df)
                X_eval_file, _, eval_meta = preprocess_data(eval_df)

                y_train_raw = train_meta.get("y_raw")
                y_eval_raw = eval_meta.get("y_raw")
                if y_train_raw is not None and y_eval_raw is not None:
                    class_names = sorted(pd.Series(y_train_raw).astype(str).str.strip().unique().tolist())
                    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

                    y_train = (
                        pd.Series(y_train_raw).astype(str).str.strip().map(class_to_idx).dropna().astype(int).reset_index(drop=True)
                    )
                    train_mask = (
                        pd.Series(y_train_raw).astype(str).str.strip().isin(class_to_idx).reset_index(drop=True)
                    )
                    X_train_file = X_train_file[train_mask.values].reset_index(drop=True)

                    eval_series = pd.Series(y_eval_raw).astype(str).str.strip()
                    eval_mask = eval_series.isin(class_to_idx)
                    y_eval = eval_series[eval_mask].map(class_to_idx).astype(int).reset_index(drop=True)
                    X_eval = X_eval_file[eval_mask.values].reset_index(drop=True)

                    X = X_train_file
                    y = y_train
                    inference_df = eval_df
                    inference_X = X_eval
                    report_true_labels = eval_series[eval_mask].reset_index(drop=True)
                    file_type = f"Eval_{detect_file_type(eval_file, eval_df)}"

        model_bundle, train_metrics = train_model(X, y, class_names=class_names, X_eval=X_eval, y_eval=y_eval)
    else:
        print("[INFO] Loading existing model...")
        model_bundle = load_model()
        train_metrics = None

    _, probs, predicted_labels = predict(model_bundle, inference_X)
    traffic_summary = analyze_traffic_patterns(inference_df, predicted_labels)
    rules_summary = simulate_detection_rules(inference_df)

    chart_path = output.save_visualization(predicted_labels)
    report_path = output.save_report(
        predicted_labels=predicted_labels,
        probabilities=probs,
        true_labels=report_true_labels,
        traffic_analysis=traffic_summary,
        rule_simulation=rules_summary,
        train_metrics=train_metrics,
    )

    print("\n====== SECURITY ANALYST OUTPUT ======")
    print(f"Input File: {file_path if not use_all_data else 'data/*.csv'}")
    print(f"Detected Type: {file_type}")
    print(f"Visualization: {chart_path}")
    print(f"Report: {report_path}")
    print(f"Traffic Pattern Summary: {traffic_summary}")
    print(f"Rule Simulation Summary: {rules_summary}")
    print("=====================================")


if __name__ == "__main__":
    dual_mode = input("Run dual strict evaluation (Friday<->Monday)? (y/n): ").strip().lower() == "y"
    if dual_mode:
        run_dual_evaluation()
        raise SystemExit(0)

    all_mode = input("Use all files from data/ ? (y/n): ").strip().lower() == "y"
    dataset_path = ""
    if not all_mode:
        dataset_path = input("Select file path (CSV): ").strip()
    mode = input("Retrain model? (y/n): ").strip().lower()
    run_pipeline(dataset_path, retrain=(mode == "y"), use_all_data=all_mode)