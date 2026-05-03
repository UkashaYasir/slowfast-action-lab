import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
from torch import nn


logger = logging.getLogger("TrainingLab")

PILOT_LABELS = ("clap", "wave", "punch", "talk", "walk")
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".webm", ".mkv", ".wmv", ".flv", ".m4v"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PilotHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(p=0.2),
            nn.Linear(input_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class BasicActionsPilotModel:
    def __init__(self, checkpoint_path: str, feature_engine=None):
        payload = torch.load(checkpoint_path, map_location="cpu")
        self.checkpoint_path = str(checkpoint_path)
        self.label_names = payload["label_names"]
        self.input_dim = int(payload["input_dim"])
        self.device = "cpu"
        self.model_name = payload.get("model_name", f"Basic Actions Pilot ({Path(checkpoint_path).stem})")
        self.feature_engine = feature_engine
        self.head = PilotHead(self.input_dim, len(self.label_names))
        self.head.load_state_dict(payload["state_dict"])
        self.head.eval()

    def predict(self, video_path: str, top_k: int = 5, feature_engine=None) -> dict:
        active_engine = feature_engine or self.feature_engine
        if active_engine is None:
            return {
                "success": False,
                "predictions": [],
                "top_prediction": None,
                "top_confidence": 0,
                "device": self.device,
                "preprocessing": {},
                "error": "Feature backbone is unavailable for the pilot model.",
            }

        embedding_result = active_engine.extract_embedding(video_path)
        if not embedding_result.get("success"):
            return {
                "success": False,
                "predictions": [],
                "top_prediction": None,
                "top_confidence": 0,
                "device": self.device,
                "preprocessing": embedding_result.get("preprocessing", {}),
                "error": embedding_result.get("error", "Embedding extraction failed."),
            }

        embedding = embedding_result["embedding"].float().unsqueeze(0)
        with torch.no_grad():
            logits = self.head(embedding)
            probs = torch.softmax(logits, dim=1)
            topk = probs.topk(k=min(top_k, probs.shape[1]))

        predictions = []
        for idx_tensor, score_tensor in zip(topk.indices[0], topk.values[0]):
            idx = int(idx_tensor.item())
            predictions.append(
                {
                    "label": self.label_names[idx],
                    "confidence": round(float(score_tensor.item()) * 100, 2),
                }
            )

        return {
            "success": True,
            "predictions": predictions,
            "top_prediction": predictions[0]["label"] if predictions else None,
            "top_confidence": predictions[0]["confidence"] if predictions else 0,
            "device": self.device,
            "preprocessing": embedding_result.get("preprocessing", {}),
            "error": None,
        }


@dataclass
class DatasetItem:
    path: str
    label: str
    split: str


class TrainingRunManager:
    def __init__(self, base_dir: str = "data/training_lab"):
        self.base_dir = Path(base_dir)
        self.dataset_dir = self.base_dir / "basic_actions_dataset"
        self.runs_dir = self.base_dir / "runs"
        self.lock = threading.Lock()
        self.current_run_id: Optional[str] = None
        self._ensure_scaffold()

    def _ensure_scaffold(self) -> None:
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        for label in PILOT_LABELS:
            (self.dataset_dir / label).mkdir(parents=True, exist_ok=True)

    def dataset_summary(self) -> dict:
        counts = {}
        total = 0
        for label in PILOT_LABELS:
            label_dir = self.dataset_dir / label
            count = sum(1 for path in label_dir.glob("*") if path.suffix.lower() in VIDEO_EXTENSIONS)
            counts[label] = count
            total += count
        return {
            "path": str(self.dataset_dir),
            "classes": list(PILOT_LABELS),
            "counts": counts,
            "total_clips": total,
            "ready_for_training": all(counts[label] >= 5 for label in PILOT_LABELS),
        }

    def _run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    def _status_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "status.json"

    def _logs_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "logs.txt"

    def _checkpoint_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "basic_actions_pilot.pt"

    def _read_run(self, run_id: str) -> Optional[dict]:
        path = self._status_path(run_id)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _write_run(self, run: dict) -> None:
        run_dir = self._run_dir(run["id"])
        run_dir.mkdir(parents=True, exist_ok=True)
        with self._status_path(run["id"]).open("w", encoding="utf-8") as file:
            json.dump(run, file, ensure_ascii=True, indent=2)

    def _append_log(self, run: dict, message: str) -> None:
        stamped = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
        run.setdefault("recent_logs", []).append(stamped)
        run["recent_logs"] = run["recent_logs"][-50:]
        with self._logs_path(run["id"]).open("a", encoding="utf-8") as file:
            file.write(stamped + "\n")
        self._write_run(run)

    def list_runs(self, limit: int = 20) -> list[dict]:
        runs = []
        for path in self.runs_dir.glob("*/status.json"):
            try:
                with path.open("r", encoding="utf-8") as file:
                    runs.append(json.load(file))
            except Exception:
                continue
        runs.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return runs[:limit]

    def get_run(self, run_id: str) -> Optional[dict]:
        return self._read_run(run_id)

    def get_current_run(self) -> Optional[dict]:
        if not self.current_run_id:
            return None
        return self._read_run(self.current_run_id)

    def get_latest_checkpoint(self) -> Optional[dict]:
        for run in self.list_runs(limit=50):
            checkpoint_path = run.get("checkpoint_path")
            if checkpoint_path and Path(checkpoint_path).exists():
                return {
                    "run_id": run["id"],
                    "run_name": run.get("name"),
                    "checkpoint_path": checkpoint_path,
                    "metrics": run.get("summary", {}),
                    "created_at": run.get("finished_at") or run.get("created_at"),
                }
        return None

    def overview(self, current_model: dict) -> dict:
        return {
            "dataset": self.dataset_summary(),
            "current_run": self.get_current_run(),
            "latest_run": self.list_runs(limit=1)[0] if self.list_runs(limit=1) else None,
            "latest_checkpoint": self.get_latest_checkpoint(),
            "current_model": current_model,
            "run_count": len(self.list_runs(limit=200)),
        }

    def start_preset_run(self, feature_engine) -> dict:
        with self.lock:
            active = self.get_current_run()
            if active and active.get("status") in {"queued", "running"}:
                raise RuntimeError("A training run is already in progress.")

            summary = self.dataset_summary()
            if not summary["ready_for_training"]:
                missing = [label for label, count in summary["counts"].items() if count < 5]
                raise RuntimeError(
                    "Dataset is not ready. Add at least 5 clips to each class folder before training. "
                    f"Missing: {', '.join(missing)}"
                )

            run_id = uuid.uuid4().hex
            run = {
                "id": run_id,
                "name": f"Basic Actions Pilot {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "preset": "basic-actions-pilot",
                "status": "queued",
                "stage": "queued",
                "created_at": utc_now(),
                "started_at": None,
                "finished_at": None,
                "current_epoch": 0,
                "total_epochs": 12,
                "dataset": summary,
                "classes": list(PILOT_LABELS),
                "recent_logs": [],
                "metrics_history": [],
                "summary": {},
                "checkpoint_path": None,
                "promoted_at": None,
                "failure_reason": None,
            }
            self.current_run_id = run_id
            self._write_run(run)
            thread = threading.Thread(
                target=self._execute_run,
                args=(run_id, feature_engine),
                daemon=True,
            )
            thread.start()
            return run

    def _build_split(self, summary: dict) -> list[DatasetItem]:
        import random

        items = []
        rng = random.Random(7)
        for label in PILOT_LABELS:
            paths = [
                path
                for path in (self.dataset_dir / label).glob("*")
                if path.suffix.lower() in VIDEO_EXTENSIONS
            ]
            rng.shuffle(paths)
            count = len(paths)
            if count < 5:
                raise RuntimeError(f"Class '{label}' needs at least 5 clips.")
            test_count = max(1, round(count * 0.2))
            val_count = max(1, round(count * 0.2))
            train_count = count - test_count - val_count
            if train_count < 2:
                train_count = 2
                val_count = max(1, count - train_count - test_count)
            train_paths = paths[:train_count]
            val_paths = paths[train_count : train_count + val_count]
            test_paths = paths[train_count + val_count :]
            items.extend(DatasetItem(str(path), label, "train") for path in train_paths)
            items.extend(DatasetItem(str(path), label, "val") for path in val_paths)
            items.extend(DatasetItem(str(path), label, "test") for path in test_paths)
        return items

    def _extract_features(self, items: list[DatasetItem], run: dict, feature_engine) -> tuple[torch.Tensor, torch.Tensor, list[DatasetItem]]:
        label_to_idx = {label: idx for idx, label in enumerate(PILOT_LABELS)}
        embeddings = []
        labels = []
        usable_items = []
        total = len(items)
        for index, item in enumerate(items, start=1):
            self._append_log(run, f"Extracting feature {index}/{total}: {Path(item.path).name}")
            result = feature_engine.extract_embedding(item.path)
            if not result.get("success"):
                self._append_log(run, f"Skipped {Path(item.path).name}: {result.get('error')}")
                continue
            embeddings.append(result["embedding"].float())
            labels.append(label_to_idx[item.label])
            usable_items.append(item)
        if not embeddings:
            raise RuntimeError("No usable training clips were found after feature extraction.")
        return torch.stack(embeddings), torch.tensor(labels, dtype=torch.long), usable_items

    def _evaluate(self, model: nn.Module, features: torch.Tensor, labels: torch.Tensor) -> dict:
        if features.numel() == 0:
            return {"loss": None, "accuracy": None}
        model.eval()
        with torch.no_grad():
            logits = model(features)
            loss = torch.nn.functional.cross_entropy(logits, labels).item()
            preds = logits.argmax(dim=1)
            accuracy = float((preds == labels).float().mean().item() * 100.0)
        return {"loss": round(loss, 4), "accuracy": round(accuracy, 2)}

    def _execute_run(self, run_id: str, feature_engine) -> None:
        run = self._read_run(run_id)
        if not run:
            return

        try:
            run["status"] = "running"
            run["stage"] = "preparing_data"
            run["started_at"] = utc_now()
            self._write_run(run)
            self._append_log(run, "Preparing dataset split.")
            items = self._build_split(run["dataset"])

            run["stage"] = "extracting_features"
            self._write_run(run)
            features, labels, usable_items = self._extract_features(items, run, feature_engine)

            split_map = {"train": [], "val": [], "test": []}
            for idx, item in enumerate(usable_items):
                split_map[item.split].append(idx)

            for split_name in ("train", "val", "test"):
                if not split_map[split_name]:
                    raise RuntimeError(f"No usable {split_name} samples remained after feature extraction.")

            train_x = features[split_map["train"]]
            train_y = labels[split_map["train"]]
            val_x = features[split_map["val"]]
            val_y = labels[split_map["val"]]
            test_x = features[split_map["test"]]
            test_y = labels[split_map["test"]]

            run["dataset"]["usable_after_extraction"] = len(usable_items)
            run["dataset"]["split_counts"] = {name: len(split_map[name]) for name in split_map}
            self._write_run(run)

            run["stage"] = "training_head"
            self._write_run(run)
            input_dim = int(train_x.shape[1])
            model = PilotHead(input_dim, len(PILOT_LABELS))
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

            best_state = None
            best_val_acc = -1.0
            stagnation = 0

            for epoch in range(1, run["total_epochs"] + 1):
                model.train()
                optimizer.zero_grad()
                logits = model(train_x)
                loss = torch.nn.functional.cross_entropy(logits, train_y)
                loss.backward()
                optimizer.step()

                train_preds = logits.argmax(dim=1)
                train_acc = float((train_preds == train_y).float().mean().item() * 100.0)
                val_metrics = self._evaluate(model, val_x, val_y)
                metrics = {
                    "epoch": epoch,
                    "train_loss": round(float(loss.item()), 4),
                    "train_accuracy": round(train_acc, 2),
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                }
                run["current_epoch"] = epoch
                run.setdefault("metrics_history", []).append(metrics)
                self._write_run(run)
                self._append_log(
                    run,
                    f"Epoch {epoch}/{run['total_epochs']} - "
                    f"train_loss {metrics['train_loss']}, train_acc {metrics['train_accuracy']}%, "
                    f"val_acc {metrics['val_accuracy']}%",
                )

                if val_metrics["accuracy"] is not None and val_metrics["accuracy"] > best_val_acc:
                    best_val_acc = val_metrics["accuracy"]
                    best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                    stagnation = 0
                else:
                    stagnation += 1
                    if stagnation >= 3:
                        self._append_log(run, "Early stopping triggered after validation stagnation.")
                        break

            if best_state is None:
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

            run["stage"] = "validating"
            self._write_run(run)
            model.load_state_dict(best_state)
            val_summary = self._evaluate(model, val_x, val_y)
            test_summary = self._evaluate(model, test_x, test_y)

            run["stage"] = "saving_checkpoint"
            checkpoint_payload = {
                "model_type": "basic_actions_pilot",
                "model_name": f"Basic Actions Pilot ({run_id[:8]})",
                "created_at": utc_now(),
                "run_id": run_id,
                "label_names": list(PILOT_LABELS),
                "input_dim": input_dim,
                "state_dict": best_state,
                "summary": {
                    "val_accuracy": val_summary["accuracy"],
                    "test_accuracy": test_summary["accuracy"],
                    "usable_samples": len(usable_items),
                },
            }
            checkpoint_path = self._checkpoint_path(run_id)
            torch.save(checkpoint_payload, checkpoint_path)

            run["checkpoint_path"] = str(checkpoint_path)
            run["summary"] = {
                "best_val_accuracy": best_val_acc,
                "final_val_accuracy": val_summary["accuracy"],
                "test_accuracy": test_summary["accuracy"],
                "usable_samples": len(usable_items),
                "train_samples": len(split_map["train"]),
                "val_samples": len(split_map["val"]),
                "test_samples": len(split_map["test"]),
            }
            run["stage"] = "completed"
            run["status"] = "completed"
            run["finished_at"] = utc_now()
            self._append_log(run, f"Training completed. Test accuracy: {test_summary['accuracy']}%.")
            self._write_run(run)
        except Exception as exc:
            run["status"] = "failed"
            run["stage"] = "failed"
            run["finished_at"] = utc_now()
            run["failure_reason"] = str(exc)
            self._append_log(run, f"Training failed: {exc}")
            self._write_run(run)
        finally:
            with self.lock:
                if self.current_run_id == run_id:
                    self.current_run_id = None

    def promote_run(self, run_id: str, registry: dict) -> dict:
        run = self._read_run(run_id)
        if not run:
            raise RuntimeError("Training run not found.")
        if run.get("status") != "completed":
            raise RuntimeError("Only completed runs can be promoted.")
        checkpoint_path = Path(run.get("checkpoint_path") or "")
        if not checkpoint_path.exists():
            raise RuntimeError("Checkpoint file is missing for this run.")

        model_id = f"basic-actions-pilot:{run_id}"
        entry = {
            "id": model_id,
            "name": f"Basic Actions Pilot ({run_id[:8]})",
            "source": "basic_actions_pilot",
            "model_type": "basic_actions_pilot",
            "checkpoint_path": str(checkpoint_path),
            "selected_at": utc_now(),
            "notes": "Promoted manually from Training Lab.",
            "classes": list(PILOT_LABELS),
            "run_id": run_id,
            "summary": run.get("summary", {}),
        }

        available = [item for item in registry.get("available_models", []) if item.get("id") != model_id]
        available.append(entry)
        registry["available_models"] = available
        registry["current_model"] = entry

        run["promoted_at"] = utc_now()
        self._write_run(run)
        return entry
