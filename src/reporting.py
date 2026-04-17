from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd


class OutputManager:
    def __init__(self, report_dir: str = "report") -> None:
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def save_visualization(self, predicted_labels, filename: str = "attack_results.png") -> str:
        labels = pd.Series(predicted_labels)
        counts = labels.value_counts()

        plt.figure(figsize=(7, 4))
        counts.plot(kind="bar")
        plt.title("Attack Classification Results")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.tight_layout()

        chart_path = self.report_dir / filename
        plt.savefig(chart_path, dpi=120)
        plt.close()
        return str(chart_path)

    def save_report(
        self,
        predicted_labels,
        probabilities=None,
        true_labels: Optional[pd.Series] = None,
        traffic_analysis: Optional[Dict] = None,
        rule_simulation: Optional[Dict] = None,
        train_metrics: Optional[Dict] = None,
        filename: str = "security_report.txt",
    ) -> str:
        total = len(predicted_labels)
        label_counts = pd.Series(predicted_labels).value_counts()
        attack_keywords = ("ddos", "dos", "brute", "attack", "portscan", "infiltration", "bot")
        attacks = int(sum(v for k, v in label_counts.items() if any(w in str(k).lower() for w in attack_keywords)))
        normal = int(total - attacks)
        attack_ratio = (attacks / total * 100.0) if total else 0.0
        avg_conf = float(probabilities.max(axis=1).mean()) if probabilities is not None else 0.0

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_text = [
            "SECURITY ANALYST REPORT",
            "=======================",
            f"Generated At: {ts}",
            f"Total Samples: {total}",
            f"Detected Attacks: {attacks}",
            f"Normal Traffic: {normal}",
            f"Attack Ratio: {attack_ratio:.2f}%",
            f"Average Model Confidence: {avg_conf:.4f}",
            "",
            "Predicted Classes:",
        ]
        report_text.extend([f"- {k}: {int(v)}" for k, v in label_counts.items()])

        if true_labels is not None:
            true_labels = pd.Series(true_labels)
            accuracy = (true_labels.values == pd.Series(predicted_labels).values).mean()
            report_text.append(f"Approx Accuracy Against Provided Labels: {accuracy:.4f}")

        if train_metrics is not None:
            class_report = train_metrics.get("classification_report", {})
            eval_total = train_metrics.get("eval_total_rows")
            eval_known = train_metrics.get("eval_known_rows")
            eval_unknown = train_metrics.get("eval_unknown_rows")
            eval_unknown_classes = train_metrics.get("eval_unknown_classes", [])
            report_text.extend(
                [
                    "",
                    "Training Metrics:",
                    f"- Split Mode: {train_metrics.get('split_mode', 'unknown')}",
                    f"- Train Samples: {train_metrics.get('train_samples', 0)}",
                    f"- Test Samples: {train_metrics.get('test_samples', 0)}",
                    f"- Validation Accuracy: {train_metrics.get('accuracy', 0.0):.4f}",
                    f"- Confusion Matrix: {train_metrics.get('confusion_matrix', [])}",
                ]
            )
            if eval_total is not None:
                coverage = (eval_known / eval_total * 100.0) if eval_total else 0.0
                report_text.extend(
                    [
                        f"- Eval Coverage (Known Classes): {eval_known}/{eval_total} ({coverage:.2f}%)",
                        f"- Eval Unseen-Class Rows: {eval_unknown}",
                        f"- Eval Unseen Classes: {eval_unknown_classes}",
                    ]
                )
            if class_report:
                report_text.append("- Per-Class Metrics:")
                for class_name, stats in class_report.items():
                    if class_name in {"accuracy", "macro avg", "weighted avg"}:
                        continue
                    if isinstance(stats, dict):
                        report_text.append(
                            f"  - {class_name}: precision={stats.get('precision', 0.0):.4f}, "
                            f"recall={stats.get('recall', 0.0):.4f}, f1={stats.get('f1-score', 0.0):.4f}, "
                            f"support={int(stats.get('support', 0))}"
                        )

        if traffic_analysis is not None:
            report_text.extend(
                [
                    "",
                    "Traffic Pattern Analysis:",
                    f"- Total Flows: {traffic_analysis.get('total_flows', 0)}",
                    f"- Average Flow Duration: {traffic_analysis.get('avg_flow_duration', 0.0):.2f}",
                    f"- Top Destination Ports: {traffic_analysis.get('top_destination_ports', {})}",
                ]
            )

        if rule_simulation is not None:
            report_text.extend(
                [
                    "",
                    "Detection Rules Simulation:",
                    f"- Rule Hits: {rule_simulation.get('rule_hits', {})}",
                    f"- Total Rule-Flagged Events: {rule_simulation.get('rule_flagged_total', 0)}",
                ]
            )

        report_path = self.report_dir / filename
        report_path.write_text("\n".join(report_text), encoding="utf-8")
        return str(report_path)
