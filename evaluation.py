"""Local evaluation helpers for NeuralVision AI."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional


EVALUATION_MANIFEST_PATH = Path(
    os.getenv("EVALUATION_MANIFEST_PATH", "data/evaluation/manifest.jsonl")
)
EVALUATION_REPORT_DIR = Path(
    os.getenv("EVALUATION_REPORT_DIR", "data/evaluation/reports")
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_label(value: Optional[str]) -> str:
    return str(value or "").strip().lower().replace("_", " ")


def load_manifest(manifest_path: Path | str = EVALUATION_MANIFEST_PATH) -> list[dict]:
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation manifest not found: {path}")

    samples = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            video_path = str(record.get("path", "")).strip()
            label = str(record.get("label", "")).strip()
            if not video_path or not label:
                raise ValueError(
                    f"Manifest line {line_number} must include 'path' and 'label'."
                )
            samples.append(
                {
                    "path": video_path,
                    "label": label,
                    "notes": record.get("notes", ""),
                }
            )

    if not samples:
        raise ValueError("Evaluation manifest is empty.")

    return samples


def apply_project_thresholds(
    raw_prediction: Optional[str],
    top_confidence: float,
    second_confidence: float,
    confidence_threshold: float,
    top1_min_confidence: float,
    top1_top2_gap_threshold: float,
) -> tuple[str, bool, float]:
    top_gap = float(top_confidence) - float(second_confidence)
    if not raw_prediction or float(top_confidence) < float(confidence_threshold):
        return "uncertain", True, top_gap
    if float(top_confidence) < float(top1_min_confidence):
        return "uncertain", True, top_gap
    if top_gap < float(top1_top2_gap_threshold):
        return "uncertain", True, top_gap
    return str(raw_prediction), False, top_gap


def evaluate_manifest(
    engine,
    manifest_path: Path | str = EVALUATION_MANIFEST_PATH,
    *,
    validator: Optional[Callable[[Path], dict]] = None,
    confidence_threshold: float = 25.0,
    top1_min_confidence: float = 45.0,
    top1_top2_gap_threshold: float = 8.0,
    top_k: int = 5,
) -> dict:
    samples = load_manifest(manifest_path)
    per_sample = []

    failed = 0
    evaluated = 0
    uncertain = 0
    accepted = 0
    raw_correct = 0
    accepted_correct = 0

    for sample in samples:
        sample_path = Path(sample["path"])
        expected_label = sample["label"]
        result_entry = {
            "path": str(sample_path),
            "label": expected_label,
            "notes": sample.get("notes", ""),
        }

        if not sample_path.exists():
            failed += 1
            result_entry["status"] = "missing_file"
            result_entry["error"] = "Video file does not exist."
            per_sample.append(result_entry)
            continue

        try:
            validation = validator(sample_path) if validator else None
            prediction = engine.predict(str(sample_path), top_k=top_k)
            if not prediction.get("success"):
                raise ValueError(prediction.get("error") or "Prediction failed.")

            predictions = prediction.get("predictions", [])
            raw_top = prediction.get("top_prediction")
            top_confidence = float(prediction.get("top_confidence") or 0.0)
            second_confidence = (
                float(predictions[1].get("confidence") or 0.0)
                if len(predictions) > 1
                else 0.0
            )
            final_prediction, is_uncertain, top_gap = apply_project_thresholds(
                raw_prediction=raw_top,
                top_confidence=top_confidence,
                second_confidence=second_confidence,
                confidence_threshold=confidence_threshold,
                top1_min_confidence=top1_min_confidence,
                top1_top2_gap_threshold=top1_top2_gap_threshold,
            )

            evaluated += 1
            if is_uncertain:
                uncertain += 1
            else:
                accepted += 1

            if normalize_label(raw_top) == normalize_label(expected_label):
                raw_correct += 1
            if not is_uncertain and normalize_label(final_prediction) == normalize_label(expected_label):
                accepted_correct += 1

            result_entry.update(
                {
                    "status": "evaluated",
                    "validation": validation,
                    "predictions": predictions,
                    "raw_top_prediction": raw_top,
                    "final_prediction": final_prediction,
                    "top_confidence": top_confidence,
                    "second_confidence": second_confidence,
                    "top_gap": top_gap,
                    "is_uncertain": is_uncertain,
                    "raw_top1_correct": normalize_label(raw_top) == normalize_label(expected_label),
                    "accepted_correct": (
                        not is_uncertain
                        and normalize_label(final_prediction) == normalize_label(expected_label)
                    ),
                }
            )
        except Exception as exc:
            failed += 1
            result_entry["status"] = "failed"
            result_entry["error"] = str(exc)

        per_sample.append(result_entry)

    total_samples = len(samples)
    raw_top1_accuracy = round((raw_correct / evaluated) * 100, 2) if evaluated else 0.0
    accepted_accuracy = round((accepted_correct / accepted) * 100, 2) if accepted else 0.0
    uncertainty_rate = round((uncertain / evaluated) * 100, 2) if evaluated else 0.0
    evaluated_coverage = round((evaluated / total_samples) * 100, 2) if total_samples else 0.0
    accepted_coverage = round((accepted / total_samples) * 100, 2) if total_samples else 0.0

    report = {
        "created_at": utc_now(),
        "manifest_path": str(Path(manifest_path)),
        "settings": {
            "confidence_threshold": confidence_threshold,
            "top1_min_confidence": top1_min_confidence,
            "top1_top2_gap_threshold": top1_top2_gap_threshold,
            "top_k": top_k,
        },
        "summary": {
            "total_samples": total_samples,
            "evaluated_samples": evaluated,
            "failed_samples": failed,
            "accepted_predictions": accepted,
            "uncertain_predictions": uncertain,
            "raw_top1_correct": raw_correct,
            "accepted_correct": accepted_correct,
            "raw_top1_accuracy": raw_top1_accuracy,
            "accepted_accuracy": accepted_accuracy,
            "uncertainty_rate": uncertainty_rate,
            "evaluated_coverage": evaluated_coverage,
            "accepted_coverage": accepted_coverage,
        },
        "examples": {
            "mismatches": [
                {
                    "path": sample["path"],
                    "label": sample["label"],
                    "raw_top_prediction": sample.get("raw_top_prediction"),
                    "final_prediction": sample.get("final_prediction"),
                    "top_confidence": sample.get("top_confidence"),
                }
                for sample in per_sample
                if sample.get("status") == "evaluated"
                and not sample.get("raw_top1_correct")
            ][:10],
            "uncertain": [
                {
                    "path": sample["path"],
                    "label": sample["label"],
                    "raw_top_prediction": sample.get("raw_top_prediction"),
                    "top_confidence": sample.get("top_confidence"),
                    "top_gap": sample.get("top_gap"),
                }
                for sample in per_sample
                if sample.get("status") == "evaluated" and sample.get("is_uncertain")
            ][:10],
            "failures": [
                {
                    "path": sample["path"],
                    "label": sample["label"],
                    "error": sample.get("error"),
                }
                for sample in per_sample
                if sample.get("status") in {"failed", "missing_file"}
            ][:10],
        },
        "samples": per_sample,
    }
    return report


def save_report(report: dict, report_dir: Path | str = EVALUATION_REPORT_DIR) -> Path:
    path = Path(report_dir)
    path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = path / f"evaluation_{timestamp}.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=True, indent=2)
    return output_path


def load_latest_report(report_dir: Path | str = EVALUATION_REPORT_DIR) -> Optional[dict]:
    path = Path(report_dir)
    if not path.exists():
        return None

    reports = sorted(path.glob("evaluation_*.json"))
    if not reports:
        return None

    latest = reports[-1]
    with latest.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    payload["report_path"] = str(latest)
    return payload
