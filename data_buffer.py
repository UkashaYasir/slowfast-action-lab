import json
import logging
import os
import shutil
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def setup_logger(name: str = "DataBuffer") -> logging.Logger:
    """Set up and return a module logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    return logger


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class VideoMetadata:
    """Metadata for a video ingested into the future-training buffer."""

    filename: str
    timestamp: str
    source_path: str
    stored_path: str
    prediction: Optional[str] = None
    prediction_source: str = "model"
    reviewed: bool = False
    reviewed_label: Optional[str] = None
    reviewed_at: Optional[str] = None
    review_notes: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=True)


class RetrainingController:
    """
    Boundary for future model training.

    The current project is local inference-first. When a labeled dataset exists,
    this is where a training command should be launched and gated by validation
    metrics before a new checkpoint is promoted.
    """

    def __init__(self):
        self.logger = setup_logger("RetrainingController")

    def retrain_model(self, dataset_path: str) -> None:
        command = os.getenv("RETRAINING_COMMAND", "").strip()
        self.logger.info("Training dataset ready at: %s", dataset_path)
        if not command:
            self.logger.info(
                "No RETRAINING_COMMAND configured. Skipping automatic training."
            )
            return

        self.logger.info(
            "RETRAINING_COMMAND is configured, but automatic execution is disabled "
            "until a reviewed labeled dataset and validation gate are added."
        )


class DataBuffer:
    """
    Local buffer for clips that may become future training data.

    Predictions saved here are machine-generated and are not treated as ground
    truth until a human reviews them.
    """

    def __init__(self, storage_dir: str = "data/buffer", threshold: int = 1000):
        self.storage_dir = Path(storage_dir)
        self.videos_dir = self.storage_dir / "videos"
        self.metadata_file = self.storage_dir / "metadata.jsonl"
        self.threshold = threshold
        self.logger = setup_logger()

        self._initialize_storage()
        self.current_count = self._load_current_count()

        self.logger.info(
            "Initialized DataBuffer. Current samples: %s/%s",
            self.current_count,
            self.threshold,
        )

    def _initialize_storage(self) -> None:
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_file.touch(exist_ok=True)

    def _load_current_count(self) -> int:
        if not self.metadata_file.exists():
            return 0
        with self.metadata_file.open("r", encoding="utf-8") as file:
            return sum(1 for line in file if line.strip())

    def list_records(
        self,
        limit: int = 50,
        reviewed: Optional[bool] = None,
    ) -> list[dict]:
        if not self.metadata_file.exists():
            return []

        records = []
        with self.metadata_file.open("r", encoding="utf-8") as file:
            for raw_line in file:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                normalized = {
                    "filename": record.get("filename"),
                    "timestamp": record.get("timestamp"),
                    "source_path": record.get("source_path", ""),
                    "stored_path": record.get("stored_path") or record.get("source_path", ""),
                    "prediction": record.get("prediction"),
                    "prediction_source": record.get("prediction_source", "model"),
                    "reviewed": bool(record.get("reviewed", False)),
                    "reviewed_label": record.get("reviewed_label"),
                    "reviewed_at": record.get("reviewed_at"),
                    "review_notes": record.get("review_notes"),
                }
                if reviewed is None or normalized["reviewed"] == reviewed:
                    records.append(normalized)

        records.sort(key=lambda item: item.get("timestamp", ""), reverse=True)
        return records[:limit]

    def mark_reviewed(
        self,
        filename: str,
        label: str,
        notes: str = "",
    ) -> Optional[dict]:
        if not self.metadata_file.exists():
            return None

        updated = None
        rewritten_lines = []
        with self.metadata_file.open("r", encoding="utf-8") as file:
            for raw_line in file:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    rewritten_lines.append(raw_line)
                    continue

                if record.get("filename") == filename:
                    record["reviewed"] = True
                    record["reviewed_label"] = label
                    record["reviewed_at"] = utc_now()
                    record["review_notes"] = notes or ""
                    updated = record
                rewritten_lines.append(json.dumps(record, ensure_ascii=True) + "\n")

        if updated is None:
            return None

        with self.metadata_file.open("w", encoding="utf-8") as file:
            file.writelines(rewritten_lines)

        return updated

    def add_video(
        self,
        source: str,
        prediction: Optional[str] = None,
        prediction_source: str = "model",
        reviewed: bool = False,
    ) -> Optional[dict]:
        """
        Store a local video file or URL and append JSONL metadata.

        Returns the metadata dict on success, or None on failure.
        """
        timestamp = utc_now()
        filename = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.mp4"
        dest_path = self.videos_dir / filename

        try:
            if source.startswith(("http://", "https://")):
                self.logger.info("Downloading video from URL: %s", source)
                urllib.request.urlretrieve(source, dest_path)
            else:
                source_path = Path(source)
                if not source_path.exists():
                    self.logger.error("Source file not found: %s", source)
                    return None
                self.logger.info("Copying local video from: %s", source)
                shutil.copy2(source_path, dest_path)
        except Exception as exc:
            self.logger.error("Failed to ingest video %s: %s", source, exc)
            return None

        metadata = VideoMetadata(
            filename=filename,
            timestamp=timestamp,
            source_path=source,
            stored_path=str(dest_path),
            prediction=prediction,
            prediction_source=prediction_source,
            reviewed=reviewed,
        )
        metadata_dict = metadata.to_dict()

        with self.metadata_file.open("a", encoding="utf-8") as file:
            file.write(metadata.to_json() + "\n")

        self.current_count += 1
        self.logger.info(
            "Added %s. Total samples: %s/%s",
            filename,
            self.current_count,
            self.threshold,
        )

        self._check_threshold()
        return metadata_dict

    def _check_threshold(self) -> None:
        if self.current_count >= self.threshold:
            self._trigger_retraining()

    def _trigger_retraining(self) -> None:
        self.logger.warning(
            "Buffer threshold reached (%s samples). Preparing reviewed-training boundary.",
            self.current_count,
        )

        training_dataset_dir = (
            self.storage_dir
            / f"training_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        temp_videos = self.storage_dir / "temp_training_videos"
        temp_metadata = self.storage_dir / "temp_training_metadata.jsonl"

        self.videos_dir.rename(temp_videos)
        self.metadata_file.rename(temp_metadata)

        training_dataset_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(temp_videos), training_dataset_dir / "videos")
        shutil.move(str(temp_metadata), training_dataset_dir / "metadata.jsonl")

        self.logger.info("Moved full buffer to %s", training_dataset_dir)

        self._initialize_storage()
        self.current_count = 0
        self.logger.info("Buffer reset and ready for new samples.")

        RetrainingController().retrain_model(str(training_dataset_dir))


if __name__ == "__main__":
    print("DataBuffer is intended to be used by main.py.")
