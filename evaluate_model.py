"""Run local evaluation on a labeled manifest of videos."""

import argparse
import json
from pathlib import Path

from evaluation import (
    EVALUATION_MANIFEST_PATH,
    EVALUATION_REPORT_DIR,
    evaluate_manifest,
    save_report,
)
from inference import SlowFastInferenceEngine
from main import (
    CONFIDENCE_THRESHOLD,
    TOP1_MIN_CONFIDENCE,
    TOP1_TOP2_GAP_THRESHOLD,
    validate_video_content,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate NeuralVision AI on labeled videos.")
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(EVALUATION_MANIFEST_PATH),
        help="Path to JSONL manifest with {path, label} entries.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(EVALUATION_REPORT_DIR),
        help="Directory for evaluation reports.",
    )
    args = parser.parse_args()

    engine = SlowFastInferenceEngine()
    report = evaluate_manifest(
        engine,
        manifest_path=args.manifest,
        validator=validate_video_content,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        top1_min_confidence=TOP1_MIN_CONFIDENCE,
        top1_top2_gap_threshold=TOP1_TOP2_GAP_THRESHOLD,
    )
    report_path = save_report(report, args.output_dir)

    summary = report["summary"]
    print("\n=== Evaluation Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved report to: {Path(report_path).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
