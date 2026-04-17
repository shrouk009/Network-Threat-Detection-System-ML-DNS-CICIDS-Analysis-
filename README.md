# Network-Threat-Detection-System-ML-DNS-CICIDS-Analysis-

This project implements an end-to-end ML workflow for network attack classification, traffic pattern analysis, and rule-based detection simulation from CSV traffic data.

It follows this pipeline:

`User selects file(s) -> Data Loader -> Auto file-type detection -> Preprocessing -> ML attack classification -> Traffic pattern analysis -> Detection rules simulation -> Results visualization -> Security analyst report`

---

## Project Structure

```text
project/
|
|-- data/
|   |-- Monday-WorkingHours.pcap_ISCX.csv
|   |-- Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
|
|-- models/
|   |-- model.pkl
|
|-- src/
|   |-- loader.py
|   |-- preprocess.py
|   |-- train.py
|   |-- predict.py
|   |-- reporting.py
|   |-- traffic_analysis.py
|   |-- rules.py
|
|-- main.py
|-- gui.py
|-- report/
|   |-- attack_results.png
|   |-- attack_results_Monday_WorkingHours.png
|   |-- attack_results_Friday_WorkingHours_Afternoon_DDos.png
|   |-- security_report.txt
|   |-- security_report_Monday_WorkingHours.txt
|   |-- security_report_Friday_WorkingHours_Afternoon_DDos.txt
|-- report.md
|-- README.md
```

---

## What We Built

- **Modular pipeline** in `src/` for loading, preprocessing, training, predicting, and reporting.
- **Auto-detection of dataset type** (e.g., Monday traffic vs Friday DDoS-style file naming/columns).
- **Multiclass label handling** for attack classification (e.g., `BENIGN`, `DDoS`, etc.).
- **Model training and persistence** using `RandomForestClassifier` saved to `models/model.pkl`.
- **Prediction layer** with class predictions, class names, and confidence scores.
- **Traffic pattern analysis** for flow behavior and top destination ports.
- **Detection rules simulation** to emulate SOC-style threshold rules over flows.
- **Strict file-based evaluation** to avoid random-split leakage and produce realistic validation.
- **Dual evaluation mode** for both directions:
  - Friday -> Monday (benign stability / false positives)
  - Monday -> Friday (unseen-class behavior)
- **Coverage-aware reporting** with known-class coverage, unseen-class row count, and unseen class names.
- **Visualization output** saved under `report/` with scenario-specific filenames.
- **Analyst-ready report output** saved under `report/` with scenario-specific filenames.
- **CLI entry point** in `main.py` and a Tkinter UI in `gui.py`, both connected to the same pipeline.

---

## Module Responsibilities

- `src/loader.py`
  - Validates selected file path and CSV format.
  - Loads single dataset or all CSV files from `data/`.
  - Detects file type (`Monday`, `Friday_DDoS`, or `Unknown`).

- `src/preprocess.py`
  - Cleans column names.
  - Finds and encodes label column when available.
  - Keeps numeric features for ML.
  - Handles missing/infinite values.

- `src/train.py`
  - Builds and trains a multiclass attack-classification pipeline.
  - Persists a model bundle (pipeline + class names) to `models/model.pkl`.
  - Supports random split and strict file-based split evaluation.
  - Produces confusion matrix and per-class precision/recall/F1 metrics.
  - Checks whether model already exists.

- `src/predict.py`
  - Loads trained model bundle.
  - Runs multiclass predictions.
  - Returns encoded predictions, probabilities, and class labels.

- `src/reporting.py`
  - Generates bar-chart visualization of predicted classes.
  - Produces text reports with model metrics, confusion matrix, per-class scores, traffic analysis, rule simulation results, and unseen-class coverage fields.

- `src/traffic_analysis.py`
  - Computes traffic behavior metrics (total flows, average flow duration).
  - Extracts top destination ports.
  - Summarizes predicted class distribution.

- `src/rules.py`
  - Simulates rule-based detections over flow features.
  - Reports per-rule hit counts and total rule-flagged events.

- `main.py`
  - Orchestrates standard pipeline runs.
  - Includes dual strict evaluation mode (`Friday <-> Monday`) that writes two independent reports and charts.

---

## Requirements

Install dependencies:

```bash
python -m pip install pandas scikit-learn matplotlib joblib
```

Recommended Python version: **3.10+**

---

## How To Run (CLI)

```bash
python main.py
```

Then provide:

1. `Run dual strict evaluation (Friday<->Monday)? (y/n)`
   - `y` runs both strict scenarios automatically and exits.
   - `n` continues to normal mode prompts.
2. (Normal mode) Whether to use all files from `data/` (`y`/`n`).
3. (Normal mode) Dataset path if single-file mode is selected.
4. (Normal mode) Whether to retrain (`y`/`n`).

### Expected Outputs

- `models/model.pkl` (trained model)
- `report/attack_results*.png` (visual charts)
- `report/security_report*.txt` (security analyst reports)

Dual mode specifically creates:
- `report/security_report_Monday_WorkingHours.txt`
- `report/security_report_Friday_WorkingHours_Afternoon_DDos.txt`
- `report/attack_results_Monday_WorkingHours.png`
- `report/attack_results_Friday_WorkingHours_Afternoon_DDos.png`

---

## How To Run (GUI)

```bash
python gui.py
```

GUI provides:
- File selection
- Train + run detection button
- Run detection button
- Result logging and generated report/plot files

---

## Security Analyst Output

The generated report includes:
- Total processed samples
- Predicted class distribution
- Number of attack-like vs normal flows
- Attack ratio (%)
- Average model confidence
- Training validation accuracy
- Split mode (`random_split` or `file_based_split`)
- Confusion matrix
- Per-class metrics (precision, recall, F1, support)
- Eval coverage for known classes
- Eval unseen-class rows and unseen class names
- Traffic pattern analysis summary
- Detection rules simulation summary

---

## Interpreting Results Correctly

- High accuracy on a benign-only evaluation file does not prove attack recall.
- If `Eval Coverage (Known Classes)` is low and `Eval Unseen Classes` includes attacks (e.g., `DDoS`), then model performance on those unseen attacks is not learned yet.
- Use dual strict reports together:
  - Friday -> Monday validates false positive behavior on benign traffic.
  - Monday -> Friday exposes unseen attack classes and coverage limits.

---

## Notes

- Training requires a dataset containing a label/class target column.
- Prediction-only mode can be used with an already saved model.
- Ensure input files are CSV format and contain usable numeric network features.
