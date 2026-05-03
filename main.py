"""NeuralVision AI FastAPI backend.

Local-only video action recognition API:
- accepts uploaded/webcam clips
- runs SlowFast R50 inference when the model is available
- provides Gemini explanations when configured
- records analysis history and stores machine-labeled clips for future review
"""

import json
import logging
import os
import re
import shutil
import uuid
import csv
import io
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from chat_service import (
    ChatStore,
    export_session_markdown,
    generate_chat_response,
    gemini_chat_configured,
)
from data_buffer import DataBuffer
from evaluation import (
    EVALUATION_MANIFEST_PATH,
    EVALUATION_REPORT_DIR,
    evaluate_manifest,
    load_latest_report,
    load_manifest,
    save_report,
)
from explainability import generate_explanation, is_gemini_configured
from training_lab import BasicActionsPilotModel, TrainingRunManager


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("API")


MODEL_NAME = "SlowFast R50 (Kinetics-400)"
UPLOAD_DIR = Path(os.getenv("NV_UPLOAD_DIR", "data/uploads"))
DATA_DIR = Path(os.getenv("NV_DATA_DIR", "data"))
HISTORY_FILE = DATA_DIR / "analysis_history.jsonl"
LOCAL_SETTINGS_FILE = DATA_DIR / "local_settings.json"
THUMBNAIL_DIR = DATA_DIR / "thumbnails"
MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
MODEL_REGISTRY_FILE = MODELS_DIR / "current_model.json"
DATASETS_DIR = Path(os.getenv("DATASETS_DIR", "datasets"))
REVIEWED_DATASET_DIR = DATASETS_DIR / "reviewed"
REVIEWED_MANIFEST_FILE = REVIEWED_DATASET_DIR / "manifest.jsonl"
TRAINING_CONFIG_DIR = Path(os.getenv("TRAINING_CONFIG_DIR", "configs/training"))
TRAINING_LAB_DIR = DATA_DIR / "training_lab"

SUPPORTED_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".webm",
    ".mkv",
    ".wmv",
    ".flv",
    ".m4v",
}
SUPPORTED_MIME_TYPES = {"application/octet-stream"}
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "200"))
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)
MIN_UPLOAD_BYTES = int(os.getenv("MIN_UPLOAD_BYTES", "1024"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "25.0"))
MIN_READABLE_FRAMES = int(os.getenv("MIN_READABLE_FRAMES", "8"))
BLANK_FRAME_STD_THRESHOLD = float(os.getenv("BLANK_FRAME_STD_THRESHOLD", "3.0"))
TOP1_MIN_CONFIDENCE = float(os.getenv("TOP1_MIN_CONFIDENCE", "52.0"))
TOP1_TOP2_GAP_THRESHOLD = float(os.getenv("TOP1_TOP2_GAP_THRESHOLD", "12.0"))

inference_engine = None
buffer = None
chat_store = None
training_manager = None
active_pilot_model = None


class ConfigRequest(BaseModel):
    gemini_api_key: Optional[str] = None


class ChatRenameRequest(BaseModel):
    title: str


class ModelSelectionRequest(BaseModel):
    checkpoint_path: Optional[str] = None
    model_id: Optional[str] = None
    notes: Optional[str] = None


class ReviewSampleRequest(BaseModel):
    label: str
    notes: Optional[str] = None


def default_model_entry() -> dict:
    return {
        "id": "default-pretrained",
        "name": MODEL_NAME,
        "source": "pretrained",
        "model_type": "slowfast_kinetics400",
        "checkpoint_path": None,
        "selected_at": utc_now(),
        "notes": "Default production model using PyTorchVideo SlowFast R50.",
    }


def load_local_settings() -> dict:
    if not LOCAL_SETTINGS_FILE.exists():
        return {}
    try:
        with LOCAL_SETTINGS_FILE.open("r", encoding="utf-8-sig") as file:
            return json.load(file)
    except json.JSONDecodeError:
        logger.warning("Local settings file is invalid JSON. Ignoring persisted settings.")
        return {}


def save_local_settings(settings: dict) -> None:
    LOCAL_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOCAL_SETTINGS_FILE.open("w", encoding="utf-8") as file:
        json.dump(settings, file, ensure_ascii=True, indent=2)


def ensure_project_scaffold() -> None:
    for directory in [
        THUMBNAIL_DIR,
        CHECKPOINTS_DIR,
        REVIEWED_DATASET_DIR,
        TRAINING_CONFIG_DIR,
        TRAINING_LAB_DIR,
        EVALUATION_MANIFEST_PATH.parent,
        EVALUATION_REPORT_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    if not MODEL_REGISTRY_FILE.exists():
        with MODEL_REGISTRY_FILE.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "current_model": default_model_entry(),
                    "available_models": [default_model_entry()],
                },
                file,
                ensure_ascii=True,
                indent=2,
            )

    REVIEWED_MANIFEST_FILE.touch(exist_ok=True)
    sample_training_config = TRAINING_CONFIG_DIR / "fine_tune.example.json"
    if not sample_training_config.exists():
        with sample_training_config.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "model_backbone": "slowfast_r50",
                    "dataset_manifest": str(REVIEWED_MANIFEST_FILE),
                    "epochs": 10,
                    "batch_size": 4,
                    "notes": "Example scaffold only. Training is not wired yet.",
                },
                file,
                ensure_ascii=True,
                indent=2,
            )


def load_model_registry() -> dict:
    if not MODEL_REGISTRY_FILE.exists():
        ensure_project_scaffold()
    with MODEL_REGISTRY_FILE.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    changed = False
    if "current_model" not in payload:
        payload["current_model"] = default_model_entry()
        changed = True
    if "available_models" not in payload or not isinstance(payload.get("available_models"), list):
        payload["available_models"] = [default_model_entry()]
        changed = True
    if not any(item.get("id") == "default-pretrained" for item in payload["available_models"]):
        payload["available_models"].insert(0, default_model_entry())
        changed = True

    current = payload.get("current_model") or {}
    if "id" not in current:
        if current.get("source") == "pretrained" and not current.get("checkpoint_path"):
            payload["current_model"] = default_model_entry()
        else:
            payload["current_model"] = {
                "id": current.get("id") or f"manual:{Path(current.get('checkpoint_path') or 'custom').stem}",
                "name": current.get("name") or Path(current.get("checkpoint_path") or "custom").stem,
                "source": current.get("source") or "custom_checkpoint",
                "model_type": current.get("model_type") or "custom_checkpoint",
                "checkpoint_path": current.get("checkpoint_path"),
                "selected_at": current.get("selected_at") or utc_now(),
                "notes": current.get("notes") or "Selected manually for future production use.",
            }
        changed = True

    if changed:
        save_model_registry(payload)
    return payload


def save_model_registry(payload: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with MODEL_REGISTRY_FILE.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=True, indent=2)


def get_active_model_entry() -> dict:
    return load_model_registry().get("current_model", default_model_entry())


def refresh_active_pilot_model() -> None:
    global active_pilot_model

    current_model = get_active_model_entry()
    if current_model.get("model_type") != "basic_actions_pilot":
        active_pilot_model = None
        return

    checkpoint_path = current_model.get("checkpoint_path")
    if not checkpoint_path or not Path(checkpoint_path).exists():
        logger.warning("Active pilot checkpoint is missing: %s", checkpoint_path)
        active_pilot_model = None
        return

    try:
        active_pilot_model = BasicActionsPilotModel(checkpoint_path, inference_engine)
    except Exception as exc:
        logger.error("Failed to load active pilot model: %s", exc)
        active_pilot_model = None


def get_active_predictor():
    current_model = get_active_model_entry()
    if current_model.get("model_type") == "basic_actions_pilot" and active_pilot_model is not None:
        return active_pilot_model, current_model
    return inference_engine, current_model


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_upload_name(original_name: Optional[str]) -> str:
    base_name = Path(original_name or "video.mp4").name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", base_name).strip("._")
    if not cleaned:
        cleaned = "video.mp4"
    return f"{uuid.uuid4().hex[:12]}_{cleaned[:120]}"


def append_history(record: dict) -> None:
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=True) + "\n")


def load_recent_analyses(limit: int = 12) -> list[dict]:
    if not HISTORY_FILE.exists():
        return []

    analyses: dict[str, dict] = {}
    ordered_ids: list[str] = []

    with HISTORY_FILE.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            record_type = record.get("type")
            analysis_id = record.get("analysis_id")
            if not analysis_id:
                continue

            if record_type == "analysis":
                analyses[analysis_id] = {
                    "analysis_id": analysis_id,
                    "timestamp": record.get("timestamp"),
                    "filename": record.get("filename"),
                    "mode": record.get("mode"),
                    "device": record.get("device"),
                    "model_name": record.get("model_name"),
                    "model_id": record.get("model_id"),
                    "model_source": record.get("model_source"),
                    "raw_top_prediction": record.get("raw_top_prediction"),
                    "top_prediction": record.get("top_prediction"),
                    "top_confidence": record.get("top_confidence"),
                    "second_confidence": record.get("second_confidence", 0.0),
                    "top_gap": record.get("top_gap", 0.0),
                    "is_uncertain": record.get("is_uncertain", False),
                    "confidence_threshold": record.get("confidence_threshold", CONFIDENCE_THRESHOLD),
                    "video_validation": record.get("video_validation"),
                    "preprocessing": record.get("preprocessing", {}),
                    "predictions": record.get("predictions", []),
                    "explanation_status": record.get("explanation_status", "not_requested"),
                    "buffer_path": record.get("buffer_path"),
                    "thumbnail_url": (
                        f"/analysis/history/{analysis_id}/thumbnail"
                        if record.get("buffer_path")
                        else None
                    ),
                }
                if analysis_id in ordered_ids:
                    ordered_ids.remove(analysis_id)
                ordered_ids.append(analysis_id)
            elif record_type == "explanation" and analysis_id in analyses:
                analyses[analysis_id]["explanation_status"] = record.get(
                    "explanation_status",
                    analyses[analysis_id].get("explanation_status", "not_requested"),
                )

    recent_ids = list(reversed(ordered_ids[-limit:]))
    return [analyses[item_id] for item_id in recent_ids if item_id in analyses]


def delete_analysis_records(analysis_id: str) -> bool:
    if not HISTORY_FILE.exists():
        return False

    kept_lines = []
    removed = False
    with HISTORY_FILE.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                kept_lines.append(raw_line)
                continue

            if record.get("analysis_id") == analysis_id:
                removed = True
                continue
            kept_lines.append(raw_line)

    if not removed:
        return False

    with HISTORY_FILE.open("w", encoding="utf-8") as file:
        file.writelines(kept_lines)
    return True


def load_analysis_record(analysis_id: str) -> Optional[dict]:
    if not HISTORY_FILE.exists():
        return None
    latest = None
    with HISTORY_FILE.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("analysis_id") == analysis_id and record.get("type") == "analysis":
                latest = record
    return latest


def ensure_analysis_thumbnail(analysis_id: str) -> Path:
    record = load_analysis_record(analysis_id)
    if not record:
        raise FileNotFoundError("Analysis record not found.")

    buffer_path = record.get("buffer_path")
    if not buffer_path:
        raise FileNotFoundError("No stored clip is available for this analysis.")

    source_path = Path(buffer_path)
    if not source_path.exists():
        raise FileNotFoundError("Stored clip is no longer available.")

    THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)
    thumbnail_path = THUMBNAIL_DIR / f"{analysis_id}.jpg"
    if thumbnail_path.exists():
        return thumbnail_path

    try:
        import cv2
    except ImportError as exc:
        raise FileNotFoundError("OpenCV is unavailable for thumbnail generation.") from exc

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise FileNotFoundError("Could not open stored clip for thumbnail generation.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target_frame = max(total_frames // 2, 0)
    if target_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise FileNotFoundError("Could not decode a thumbnail frame.")

    cv2.imwrite(str(thumbnail_path), frame)
    return thumbnail_path


def export_analysis_history_payload(limit: int = 500) -> list[dict]:
    return load_recent_analyses(limit)


def export_analysis_history_csv(limit: int = 500) -> str:
    rows = export_analysis_history_payload(limit)
    fieldnames = [
        "analysis_id",
        "timestamp",
        "filename",
        "model_name",
        "model_id",
        "top_prediction",
        "raw_top_prediction",
        "top_confidence",
        "second_confidence",
        "top_gap",
        "is_uncertain",
        "mode",
        "device",
        "explanation_status",
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key) for key in fieldnames})
    return output.getvalue()


def upsert_reviewed_manifest_entry(record: dict) -> dict:
    ensure_project_scaffold()

    entry = {
        "path": record.get("stored_path"),
        "label": record.get("reviewed_label") or record.get("prediction"),
        "notes": record.get("review_notes", ""),
        "source_buffer_filename": record.get("filename"),
        "source_prediction": record.get("prediction"),
        "reviewed_at": record.get("reviewed_at") or utc_now(),
    }

    existing_entries = []
    replaced = False
    if REVIEWED_MANIFEST_FILE.exists():
        with REVIEWED_MANIFEST_FILE.open("r", encoding="utf-8") as file:
            for raw_line in file:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if item.get("source_buffer_filename") == entry["source_buffer_filename"]:
                    existing_entries.append(entry)
                    replaced = True
                else:
                    existing_entries.append(item)

    if not replaced:
        existing_entries.append(entry)

    with REVIEWED_MANIFEST_FILE.open("w", encoding="utf-8") as file:
        for item in existing_entries:
            file.write(json.dumps(item, ensure_ascii=True) + "\n")

    return entry


def load_review_samples(limit: int = 50, status: str = "pending") -> dict:
    if not buffer:
        return {"items": [], "counts": {"pending": 0, "reviewed": 0, "total": 0}}

    all_items = buffer.list_records(limit=1000, reviewed=None)
    pending_count = sum(1 for item in all_items if not item.get("reviewed"))
    reviewed_count = sum(1 for item in all_items if item.get("reviewed"))

    if status == "reviewed":
        filtered = [item for item in all_items if item.get("reviewed")]
    elif status == "all":
        filtered = all_items
    else:
        filtered = [item for item in all_items if not item.get("reviewed")]

    items = []
    for item in filtered[:limit]:
        filename = item.get("filename")
        items.append(
            {
                **item,
                "video_url": f"/review/samples/{filename}/video" if filename else None,
                "thumbnail_url": f"/review/samples/{filename}/thumbnail" if filename else None,
            }
        )

    return {
        "items": items,
        "counts": {
            "pending": pending_count,
            "reviewed": reviewed_count,
            "total": len(all_items),
        },
    }


def get_review_sample(filename: str) -> Optional[dict]:
    samples = load_review_samples(limit=1000, status="all")["items"]
    for item in samples:
        if item.get("filename") == filename:
            return item
    return None


def ensure_review_thumbnail(filename: str) -> Path:
    sample = get_review_sample(filename)
    if not sample:
        raise FileNotFoundError("Review sample not found.")

    source_path = Path(sample.get("stored_path") or "")
    if not source_path.exists():
        raise FileNotFoundError("Stored review clip is not available.")

    THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)
    thumbnail_path = THUMBNAIL_DIR / f"review_{filename}.jpg"
    if thumbnail_path.exists():
        return thumbnail_path

    try:
        import cv2
    except ImportError as exc:
        raise FileNotFoundError("OpenCV is unavailable for thumbnail generation.") from exc

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise FileNotFoundError("Could not open review clip for thumbnail generation.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(total_frames // 2, 0))
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise FileNotFoundError("Could not decode thumbnail frame.")

    cv2.imwrite(str(thumbnail_path), frame)
    return thumbnail_path


def apply_confidence_threshold(raw_prediction: Optional[str], confidence: float) -> tuple[str, bool]:
    if not raw_prediction or confidence < CONFIDENCE_THRESHOLD:
        return "uncertain", True
    return raw_prediction, False


def mock_result() -> dict:
    return {
        "success": True,
        "device": "none",
        "predictions": [
            {"label": "punch", "confidence": 89.3},
            {"label": "boxing", "confidence": 6.2},
            {"label": "slapping", "confidence": 2.1},
            {"label": "arm wrestling", "confidence": 1.4},
            {"label": "shaking hands", "confidence": 1.0},
        ],
        "top_prediction": "punch",
        "top_confidence": 89.3,
        "error": None,
    }


def run_video_analysis(file_path: Path, original_filename: Optional[str] = None) -> dict:
    """Shared analysis routine for direct uploads and chat attachments."""
    video_validation = validate_video_content(file_path)
    logger.info("Video validation passed: %s", video_validation)

    predictor, model_entry = get_active_predictor()

    if predictor is not None and inference_engine is not None:
        logger.info("Starting live inference for %s", file_path.name)
        if model_entry.get("model_type") == "basic_actions_pilot" and active_pilot_model is not None:
            result = active_pilot_model.predict(str(file_path), top_k=5)
            mode = "pilot"
        else:
            result = inference_engine.predict(str(file_path), top_k=5)
            mode = "live"
        if not result.get("success"):
            raise HTTPException(
                status_code=422,
                detail=(
                    "Inference failed: "
                    f"{result.get('error', 'unknown error')}. "
                    "Try MP4/H.264 or a clip at least 2 seconds long."
                ),
            )
    else:
        logger.warning("Using mock predictions because model is not loaded.")
        result = mock_result()
        mode = "mock"

    raw_top_prediction = result.get("top_prediction")
    top_confidence = float(result.get("top_confidence") or 0)
    top_predictions = result.get("predictions", [])
    second_confidence = float(top_predictions[1].get("confidence") or 0) if len(top_predictions) > 1 else 0.0
    top_gap = top_confidence - second_confidence

    top_prediction, is_uncertain = apply_confidence_threshold(
        raw_top_prediction,
        top_confidence,
    )
    if not is_uncertain and (
        top_confidence < TOP1_MIN_CONFIDENCE or top_gap < TOP1_TOP2_GAP_THRESHOLD
    ):
        top_prediction = "uncertain"
        is_uncertain = True

    buffer_record = None
    if buffer and file_path.exists():
        try:
            buffer_record = buffer.add_video(
                source=str(file_path),
                prediction=raw_top_prediction,
                prediction_source="model",
                reviewed=False,
            )
        except Exception as exc:
            logger.warning("Buffer add failed: %s", exc)

    return {
        "analysis_id": uuid.uuid4().hex,
        "status": "success",
        "mode": mode,
        "filename": original_filename,
        "model_name": model_entry.get("name", MODEL_NAME),
        "model_id": model_entry.get("id", "default-pretrained"),
        "model_source": model_entry.get("source", "pretrained"),
        "device": result.get("device", "none"),
        "predictions": top_predictions,
        "raw_top_prediction": raw_top_prediction,
        "top_prediction": top_prediction,
        "top_confidence": top_confidence,
        "second_confidence": second_confidence,
        "top_gap": top_gap,
        "is_uncertain": is_uncertain,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "top1_min_confidence": TOP1_MIN_CONFIDENCE,
        "top1_top2_gap_threshold": TOP1_TOP2_GAP_THRESHOLD,
        "video_validation": video_validation,
        "preprocessing": result.get("preprocessing", {}),
        "buffer_path": buffer_record.get("stored_path") if buffer_record else None,
    }


async def save_upload(video: UploadFile) -> tuple[Path, int]:
    original_name = video.filename or ""
    extension = Path(original_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid file type. Supported formats: "
                + ", ".join(sorted(SUPPORTED_EXTENSIONS))
            ),
        )

    content_type = (video.content_type or "").lower()
    if content_type and not (
        content_type.startswith("video/") or content_type in SUPPORTED_MIME_TYPES
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type: {content_type}. Upload a video file.",
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_path = UPLOAD_DIR / safe_upload_name(original_name)
    total_size = 0

    try:
        with file_path.open("wb") as output:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Video is too large. Limit is {MAX_UPLOAD_MB:g} MB.",
                    )
                output.write(chunk)
    except HTTPException:
        file_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}")

    if total_size < MIN_UPLOAD_BYTES:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded video is empty or too small.")

    return file_path, total_size


def validate_video_content(file_path: Path) -> dict:
    """
    Reject clips that are technically files but do not contain usable frames.

    Action models always output a class, even for blank video. This preflight
    keeps blank/corrupt clips from being treated as valid model input.
    """
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        logger.warning("OpenCV validation unavailable: %s", exc)
        return {"validation": "skipped", "reason": "opencv_unavailable"}

    cap = cv2.VideoCapture(str(file_path))
    if not cap.isOpened():
        raise HTTPException(
            status_code=422,
            detail="Video could not be opened. Try a standard MP4/H.264 or WEBM clip.",
        )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if width <= 0 or height <= 0:
        cap.release()
        raise HTTPException(status_code=422, detail="Video has no valid frame size.")

    max_frames_to_read = min(total_frames, int(fps * 5)) if total_frames > 0 else int(fps * 5)
    max_frames_to_read = max(max_frames_to_read, MIN_READABLE_FRAMES)

    means = []
    stds = []
    read_frames = 0

    for _ in range(max_frames_to_read):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        means.append(float(np.mean(gray)))
        stds.append(float(np.std(gray)))
        read_frames += 1

    cap.release()

    if read_frames < MIN_READABLE_FRAMES:
        raise HTTPException(
            status_code=422,
            detail=(
                "Video is too short or unreadable. Use a clip with at least "
                f"{MIN_READABLE_FRAMES} readable frames."
            ),
        )

    average_std = float(np.mean(stds)) if stds else 0.0
    average_mean = float(np.mean(means)) if means else 0.0

    if average_std < BLANK_FRAME_STD_THRESHOLD:
        raise HTTPException(
            status_code=422,
            detail=(
                "Video has no usable visual content. Frames appear blank or "
                "nearly uniform, so analysis was not run."
            ),
        )

    return {
        "validation": "passed",
        "readable_frames": read_frames,
        "reported_frames": total_frames,
        "fps": round(fps, 2),
        "width": width,
        "height": height,
        "average_brightness": round(average_mean, 2),
        "average_frame_std": round(average_std, 2),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global inference_engine, buffer, chat_store, training_manager

    del app
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    ensure_project_scaffold()

    logger.info("=" * 50)
    logger.info("NeuralVision AI Backend starting")
    logger.info("Local-only mode, model: %s", MODEL_NAME)
    logger.info("=" * 50)

    buffer = DataBuffer(storage_dir="data/buffer", threshold=1000)
    chat_store = ChatStore()
    training_manager = TrainingRunManager(base_dir=str(TRAINING_LAB_DIR))

    settings = load_local_settings()
    stored_key = (settings.get("gemini_api_key") or "").strip()
    if stored_key and not os.getenv("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = stored_key

    try:
        from inference import SlowFastInferenceEngine

        inference_engine = SlowFastInferenceEngine()
        logger.info("Inference engine loaded successfully.")
    except Exception as exc:
        logger.error("Failed to load inference engine: %s", exc)
        logger.warning("Backend will run in mock mode.")
        inference_engine = None

    refresh_active_pilot_model()

    if is_gemini_configured():
        logger.info("Gemini API key detected; explanations enabled.")
    else:
        logger.info("GEMINI_API_KEY not set; explanations will use fallback text.")

    yield

    logger.info("Backend shutting down. Cleaning temp uploads.")
    shutil.rmtree(UPLOAD_DIR, ignore_errors=True)


app = FastAPI(
    title="NeuralVision AI API",
    description="Local video intelligence API powered by SlowFast R50.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    current_model = get_active_model_entry()
    return {
        "service": "NeuralVision AI",
        "status": "online",
        "model_name": current_model.get("name", MODEL_NAME),
        "inference_mode": "pilot" if current_model.get("model_type") == "basic_actions_pilot" else "live" if inference_engine else "mock",
        "device": inference_engine.device if inference_engine else "none",
    }


@app.get("/health")
async def health():
    current_model = get_active_model_entry()
    manifest_count = 0
    try:
        manifest_count = len(load_manifest(EVALUATION_MANIFEST_PATH))
    except Exception:
        manifest_count = 0

    return {
        "status": "healthy",
        "model_loaded": inference_engine is not None,
        "model_name": current_model.get("name", MODEL_NAME),
        "device": inference_engine.device if inference_engine else "none",
        "inference_mode": (
            "pilot"
            if current_model.get("model_type") == "basic_actions_pilot" and active_pilot_model is not None
            else "live" if inference_engine else "mock"
        ),
        "gemini_configured": is_gemini_configured(),
        "buffer_count": buffer.current_count if buffer else 0,
        "buffer_threshold": buffer.threshold if buffer else 0,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "top1_min_confidence": TOP1_MIN_CONFIDENCE,
        "top1_top2_gap_threshold": TOP1_TOP2_GAP_THRESHOLD,
        "max_upload_mb": MAX_UPLOAD_MB,
        "chat_ready": True,
        "evaluation_manifest_samples": manifest_count,
        "current_model": current_model,
        "training_lab_ready": training_manager is not None,
        "training_current_run": training_manager.get_current_run() if training_manager else None,
    }


@app.get("/models/current")
async def get_current_model():
    registry = load_model_registry()
    return {
        "status": "success",
        "current_model": registry.get("current_model", {}),
        "available_models": registry.get("available_models", []),
    }


@app.post("/models/current")
async def set_current_model(payload: ModelSelectionRequest):
    registry = load_model_registry()
    if payload.model_id:
        selected = next((item for item in registry.get("available_models", []) if item.get("id") == payload.model_id), None)
        if not selected:
            raise HTTPException(status_code=404, detail="Requested model_id is not available.")
        selected = {
            **selected,
            "selected_at": utc_now(),
            "notes": payload.notes or selected.get("notes"),
        }
        registry["current_model"] = selected
    elif payload.checkpoint_path:
        checkpoint_path = Path(payload.checkpoint_path)
        if not checkpoint_path.exists():
            raise HTTPException(status_code=404, detail="Checkpoint path does not exist.")
        registry["current_model"] = {
            "id": f"manual:{checkpoint_path.stem}",
            "name": checkpoint_path.stem,
            "source": "custom_checkpoint",
            "model_type": "custom_checkpoint",
            "checkpoint_path": str(checkpoint_path),
            "selected_at": utc_now(),
            "notes": payload.notes or "Selected manually for future production use.",
        }
    else:
        raise HTTPException(status_code=400, detail="Provide model_id or checkpoint_path.")
    save_model_registry(registry)
    refresh_active_pilot_model()
    return {"status": "success", "current_model": registry["current_model"]}


@app.post("/models/current/reset")
async def reset_current_model():
    registry = load_model_registry()
    registry["current_model"] = default_model_entry()
    if not any(item.get("id") == "default-pretrained" for item in registry.get("available_models", [])):
        registry.setdefault("available_models", []).insert(0, default_model_entry())
    save_model_registry(registry)
    refresh_active_pilot_model()
    return {"status": "success", "current_model": registry["current_model"]}


@app.get("/evaluation/latest")
async def get_latest_evaluation():
    latest_report = load_latest_report(EVALUATION_REPORT_DIR)
    manifest_exists = EVALUATION_MANIFEST_PATH.exists()

    try:
        manifest_count = len(load_manifest(EVALUATION_MANIFEST_PATH)) if manifest_exists else 0
    except Exception as exc:
        manifest_count = 0
        manifest_error = str(exc)
    else:
        manifest_error = None

    return {
        "status": "success",
        "manifest_path": str(EVALUATION_MANIFEST_PATH),
        "manifest_exists": manifest_exists,
        "manifest_samples": manifest_count,
        "manifest_error": manifest_error,
        "latest_report": latest_report,
    }


@app.post("/evaluation/run")
async def run_evaluation():
    predictor, model_entry = get_active_predictor()
    if inference_engine is None or predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded, so evaluation cannot run.")
    if not EVALUATION_MANIFEST_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Evaluation manifest not found: {EVALUATION_MANIFEST_PATH}",
        )

    try:
        report = evaluate_manifest(
            predictor,
            manifest_path=EVALUATION_MANIFEST_PATH,
            validator=validate_video_content,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            top1_min_confidence=TOP1_MIN_CONFIDENCE,
            top1_top2_gap_threshold=TOP1_TOP2_GAP_THRESHOLD,
        )
        report["model"] = {
            "id": model_entry.get("id"),
            "name": model_entry.get("name"),
            "source": model_entry.get("source"),
            "model_type": model_entry.get("model_type"),
        }
        report_path = save_report(report, EVALUATION_REPORT_DIR)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Evaluation failed: {exc}") from exc

    return {
        "status": "success",
        "report_path": str(report_path),
        "summary": report.get("summary", {}),
        "created_at": report.get("created_at"),
    }


@app.get("/evaluation/latest/export")
async def export_latest_evaluation():
    latest_report = load_latest_report(EVALUATION_REPORT_DIR)
    if not latest_report:
        raise HTTPException(status_code=404, detail="No evaluation report has been generated yet.")

    output = io.BytesIO(json.dumps(latest_report, ensure_ascii=True, indent=2).encode("utf-8"))
    headers = {"Content-Disposition": 'attachment; filename="latest_evaluation_report.json"'}
    return StreamingResponse(output, media_type="application/json", headers=headers)


@app.get("/training-lab/overview")
async def get_training_lab_overview():
    if not training_manager:
        raise HTTPException(status_code=503, detail="Training Lab is not initialized.")
    return {
        "status": "success",
        "overview": training_manager.overview(get_active_model_entry()),
    }


@app.get("/training-runs")
async def list_training_runs(limit: int = 20):
    if not training_manager:
        raise HTTPException(status_code=503, detail="Training Lab is not initialized.")
    safe_limit = max(1, min(limit, 100))
    return {
        "status": "success",
        "runs": training_manager.list_runs(limit=safe_limit),
        "current_run": training_manager.get_current_run(),
    }


@app.get("/training-runs/current")
async def get_current_training_run():
    if not training_manager:
        raise HTTPException(status_code=503, detail="Training Lab is not initialized.")
    return {
        "status": "success",
        "run": training_manager.get_current_run(),
    }


@app.get("/training-runs/{run_id}")
async def get_training_run(run_id: str):
    if not training_manager:
        raise HTTPException(status_code=503, detail="Training Lab is not initialized.")
    run = training_manager.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found.")
    return {"status": "success", "run": run}


@app.post("/training-runs/start")
async def start_training_run():
    if not training_manager:
        raise HTTPException(status_code=503, detail="Training Lab is not initialized.")
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="SlowFast backbone is unavailable, so Training Lab cannot start.")
    try:
        run = training_manager.start_preset_run(inference_engine)
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"status": "success", "run": run}


@app.post("/training-runs/{run_id}/promote")
async def promote_training_run(run_id: str):
    if not training_manager:
        raise HTTPException(status_code=503, detail="Training Lab is not initialized.")
    try:
        registry = load_model_registry()
        promoted = training_manager.promote_run(run_id, registry)
        save_model_registry(registry)
        refresh_active_pilot_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "status": "success",
        "current_model": promoted,
        "available_models": load_model_registry().get("available_models", []),
    }


@app.post("/configure")
async def configure(config: ConfigRequest):
    gemini_api_key = (config.gemini_api_key or "").strip()
    if not gemini_api_key:
        return {
            "status": "no_change",
            "message": "No Gemini API key provided.",
            "gemini_configured": is_gemini_configured(),
        }

    os.environ["GEMINI_API_KEY"] = gemini_api_key
    settings = load_local_settings()
    settings["gemini_api_key"] = gemini_api_key
    save_local_settings(settings)
    logger.info("Gemini API key configured for this backend process.")
    return {
        "status": "success",
        "message": "Gemini API key configured for this session.",
        "gemini_configured": True,
    }


@app.post("/upload-video")
async def upload_video(video: UploadFile = File(...)):
    analysis_id = uuid.uuid4().hex
    file_path: Optional[Path] = None

    try:
        file_path, file_size = await save_upload(video)
        logger.info(
            "Saved upload %s (%0.2f MB)",
            file_path.name,
            file_size / (1024 * 1024),
        )
        response = run_video_analysis(file_path, video.filename)
        response["analysis_id"] = analysis_id

        append_history(
            {
                "type": "analysis",
                "analysis_id": analysis_id,
                "timestamp": utc_now(),
                "filename": video.filename,
                "stored_upload": str(file_path),
                "model_name": response["model_name"],
                "model_id": response["model_id"],
                "model_source": response["model_source"],
                "mode": response["mode"],
                "device": response["device"],
                "predictions": response["predictions"],
                "raw_top_prediction": response["raw_top_prediction"],
                "top_prediction": response["top_prediction"],
                "top_confidence": response["top_confidence"],
                "second_confidence": response["second_confidence"],
                "top_gap": response["top_gap"],
                "is_uncertain": response["is_uncertain"],
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "video_validation": response["video_validation"],
                "preprocessing": response["preprocessing"],
                "explanation_status": "not_requested",
                "buffer_path": response["buffer_path"],
                "prediction_source": "model",
                "reviewed": False,
            }
        )

        return response
    finally:
        if file_path and file_path.exists():
            try:
                file_path.unlink()
            except Exception as exc:
                logger.warning("Failed to clean temp upload %s: %s", file_path, exc)


@app.get("/explanation")
async def get_explanation(prediction: str, analysis_id: Optional[str] = None):
    prediction = (prediction or "").strip()
    if not prediction:
        raise HTTPException(status_code=400, detail="Prediction query parameter is required.")

    explanation = generate_explanation(prediction=prediction)
    explanation_status = (
        "skipped"
        if prediction.lower() == "uncertain"
        else "generated"
        if is_gemini_configured() and not explanation.startswith("Explanation unavailable")
        else "unavailable"
    )

    if analysis_id:
        append_history(
            {
                "type": "explanation",
                "analysis_id": analysis_id,
                "timestamp": utc_now(),
                "prediction": prediction,
                "explanation_status": explanation_status,
                "gemini_configured": is_gemini_configured(),
            }
        )

    return {
        "status": "success",
        "prediction": prediction,
        "explanation": explanation,
        "explanation_status": explanation_status,
        "gemini_configured": is_gemini_configured(),
    }


@app.get("/analysis/history")
async def get_analysis_history(limit: int = 12):
    safe_limit = max(1, min(limit, 50))
    return {
        "status": "success",
        "analyses": load_recent_analyses(safe_limit),
    }


@app.get("/analysis/history/export")
async def export_analysis_history(format: str = "json", limit: int = 500):
    safe_limit = max(1, min(limit, 1000))
    if format.lower() == "csv":
        csv_payload = export_analysis_history_csv(safe_limit)
        headers = {"Content-Disposition": 'attachment; filename="analysis_history.csv"'}
        return StreamingResponse(
            io.BytesIO(csv_payload.encode("utf-8")),
            media_type="text/csv",
            headers=headers,
        )

    payload = export_analysis_history_payload(safe_limit)
    headers = {"Content-Disposition": 'attachment; filename="analysis_history.json"'}
    return StreamingResponse(
        io.BytesIO(json.dumps(payload, ensure_ascii=True, indent=2).encode("utf-8")),
        media_type="application/json",
        headers=headers,
    )


@app.get("/analysis/history/{analysis_id}/thumbnail")
async def get_analysis_thumbnail(analysis_id: str):
    try:
        thumbnail_path = ensure_analysis_thumbnail(analysis_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(thumbnail_path, media_type="image/jpeg")


@app.get("/review/samples")
async def get_review_samples(limit: int = 24, status: str = "pending"):
    safe_limit = max(1, min(limit, 100))
    safe_status = status.lower()
    if safe_status not in {"pending", "reviewed", "all"}:
        raise HTTPException(status_code=400, detail="status must be pending, reviewed, or all.")

    payload = load_review_samples(safe_limit, safe_status)
    return {
        "status": "success",
        "samples": payload["items"],
        "counts": payload["counts"],
    }


@app.post("/review/samples/{filename}")
async def review_sample(filename: str, payload: ReviewSampleRequest):
    if not buffer:
        raise HTTPException(status_code=503, detail="Buffer system is not initialized.")

    label = payload.label.strip()
    if not label:
        raise HTTPException(status_code=400, detail="Reviewed label is required.")

    updated = buffer.mark_reviewed(filename, label=label, notes=(payload.notes or "").strip())
    if not updated:
        raise HTTPException(status_code=404, detail="Review sample not found.")

    manifest_entry = upsert_reviewed_manifest_entry(updated)
    return {
        "status": "success",
        "sample": updated,
        "reviewed_manifest_entry": manifest_entry,
    }


@app.get("/review/samples/{filename}/video")
async def get_review_video(filename: str):
    sample = get_review_sample(filename)
    if not sample:
        raise HTTPException(status_code=404, detail="Review sample not found.")

    video_path = Path(sample.get("stored_path") or "")
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Stored review clip is not available.")
    return FileResponse(video_path, media_type="video/mp4", filename=video_path.name)


@app.get("/review/samples/{filename}/thumbnail")
async def get_review_thumbnail(filename: str):
    try:
        thumbnail_path = ensure_review_thumbnail(filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(thumbnail_path, media_type="image/jpeg")


@app.delete("/analysis/history/{analysis_id}")
async def delete_analysis_history(analysis_id: str):
    deleted = delete_analysis_records(analysis_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Analysis entry not found.")
    return {
        "status": "success",
        "deleted_analysis_id": analysis_id,
    }


@app.get("/chat/history")
async def get_chat_history():
    sessions = chat_store.list_sessions() if chat_store else []
    return {
        "status": "success",
        "sessions": sessions,
        "gemini_configured": gemini_chat_configured(),
    }


@app.get("/chat/history/{session_id}")
async def get_chat_session(session_id: str):
    if not chat_store:
        raise HTTPException(status_code=503, detail="Chat store is not initialized.")

    session = chat_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found.")
    return {"status": "success", "session": session}


@app.patch("/chat/history/{session_id}")
async def rename_chat_session(session_id: str, payload: ChatRenameRequest):
    if not chat_store:
        raise HTTPException(status_code=503, detail="Chat store is not initialized.")

    session = chat_store.rename_session(session_id, payload.title)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found or title was empty.")
    return {"status": "success", "session": session}


@app.delete("/chat/history/{session_id}")
async def delete_chat_session(session_id: str):
    if not chat_store:
        raise HTTPException(status_code=503, detail="Chat store is not initialized.")

    deleted = chat_store.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Chat session not found.")
    return {"status": "success", "deleted_session_id": session_id}


@app.get("/chat/history/{session_id}/export")
async def export_chat_session(session_id: str):
    if not chat_store:
        raise HTTPException(status_code=503, detail="Chat store is not initialized.")

    session = chat_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found.")

    markdown = export_session_markdown(session)
    headers = {
        "Content-Disposition": f'attachment; filename="chat_{session_id}.md"'
    }
    return StreamingResponse(
        io.BytesIO(markdown.encode("utf-8")),
        media_type="text/markdown",
        headers=headers,
    )


@app.post("/chat/send")
async def send_chat_message(
    message: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    video: Optional[UploadFile] = File(None),
):
    if not chat_store:
        raise HTTPException(status_code=503, detail="Chat store is not initialized.")

    user_message = (message or "").strip()
    if not user_message and video is None:
        raise HTTPException(status_code=400, detail="Send a message or attach a video.")

    session = chat_store.ensure_session(session_id, user_message or "Video conversation")
    file_path: Optional[Path] = None
    video_context = None
    attachment = None

    try:
        if video is not None:
            file_path, file_size = await save_upload(video)
            attachment = {
                "filename": video.filename,
                "size_mb": round(file_size / (1024 * 1024), 2),
            }
            video_context = run_video_analysis(file_path, video.filename)

        assistant_reply = generate_chat_response(
            user_message=user_message,
            history=session.get("messages", []),
            video_context=video_context,
            recent_analyses=load_recent_analyses(5),
        )

        user_entry = {
            "id": uuid.uuid4().hex,
            "role": "user",
            "content": user_message or "Analyze the attached video.",
            "created_at": utc_now(),
            "attachment": attachment,
        }
        assistant_entry = {
            "id": uuid.uuid4().hex,
            "role": "assistant",
            "content": assistant_reply,
            "created_at": utc_now(),
            "video_context": video_context,
        }
        session = chat_store.append_messages(session["id"], [user_entry, assistant_entry])

        append_history(
            {
                "type": "chat",
                "session_id": session["id"],
                "timestamp": utc_now(),
                "user_message": user_entry["content"],
                "assistant_message": assistant_reply,
                "has_video": bool(video_context),
                "video_context": video_context,
                "gemini_configured": gemini_chat_configured(),
            }
        )

        return {
            "status": "success",
            "session": {
                "id": session["id"],
                "title": session.get("title", "New conversation"),
                "updated_at": session.get("updated_at"),
            },
            "messages": [user_entry, assistant_entry],
            "gemini_configured": gemini_chat_configured(),
        }
    finally:
        if file_path and file_path.exists():
            try:
                file_path.unlink()
            except Exception as exc:
                logger.warning("Failed to clean chat upload %s: %s", file_path, exc)


@app.post("/add-to-buffer")
async def add_to_buffer(source: str = Form(...), prediction: Optional[str] = Form(None)):
    if not buffer:
        raise HTTPException(status_code=503, detail="Buffer system is not initialized.")

    record = buffer.add_video(
        source=source,
        prediction=prediction,
        prediction_source="manual",
        reviewed=False,
    )
    if not record:
        raise HTTPException(status_code=500, detail="Failed to add video to buffer.")

    return {
        "status": "success",
        "message": f"Added to buffer. Current: {buffer.current_count}/{buffer.threshold}",
        "record": record,
    }


@app.get("/datasets/reviewed/manifest/export")
async def export_reviewed_manifest():
    ensure_project_scaffold()
    return FileResponse(
        REVIEWED_MANIFEST_FILE,
        media_type="application/json",
        filename="reviewed_manifest.jsonl",
    )


if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 50)
    print("NeuralVision AI - Starting Server")
    print("API:  http://localhost:8000")
    print("Docs: http://localhost:8000/docs")
    print("=" * 50 + "\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
