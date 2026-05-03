"""
SlowFast Video Inference Module
===============================
Handles model loading, video preprocessing, clip-level inference, and
backbone feature extraction for the Training Lab pilot model.
"""

import argparse
import json
import logging
import os
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("Inference")


MODEL_NAME = "SlowFast R50 (Kinetics-400)"
TORCH_HUB_DIR = Path(os.getenv("TORCH_HUB_DIR", "data/model_cache/torch_hub"))
KINETICS_LABELS_PATH = Path(os.getenv("KINETICS_LABELS_PATH", "kinetics_classnames.json"))
TARGET_CLIP_SECONDS = float(os.getenv("TARGET_CLIP_SECONDS", "4.0"))
MAX_SOURCE_SECONDS = float(os.getenv("MAX_SOURCE_SECONDS", "8.0"))
NORMALIZE_CONTAINER_FOR_INFERENCE = os.getenv("NORMALIZE_CONTAINER_FOR_INFERENCE", "1") == "1"
NORMALIZED_VIDEO_DIR = Path(os.getenv("NORMALIZED_VIDEO_DIR", "data/uploads/normalized"))
NORMALIZE_EXTENSIONS = {".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v"}


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames into a list of tensors
    [slow_pathway, fast_pathway] for the SlowFast model.
    """

    def __init__(self, alpha=4):
        super().__init__()
        self.alpha = alpha

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(0, frames.shape[1] - 1, frames.shape[1] // self.alpha).long(),
        )
        return [slow_pathway, fast_pathway]


def get_transform(num_frames=32, crop_size=256, alpha=4):
    """Returns the transformation pipeline for SlowFast."""
    return ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                ShortSideScale(size=crop_size),
                CenterCropVideo(crop_size),
                PackPathway(alpha=alpha),
            ]
        ),
    )


def download_kinetics_labels():
    """Downloads Kinetics-400 labels and returns an id-to-label dictionary."""
    url = "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
    if not KINETICS_LABELS_PATH.exists():
        logger.info("Downloading Kinetics labels to %s...", KINETICS_LABELS_PATH)
        KINETICS_LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, KINETICS_LABELS_PATH)

    with KINETICS_LABELS_PATH.open("r", encoding="utf-8") as file:
        kinetics_classnames = json.load(file)

    kinetics_id_to_classname = {}
    for label, idx in kinetics_classnames.items():
        clean_label = label.strip('"').replace('"', "").strip()
        kinetics_id_to_classname[idx] = clean_label

    logger.info("Loaded %s Kinetics-400 labels.", len(kinetics_id_to_classname))
    missing = [i for i in range(400) if i not in kinetics_id_to_classname]
    if missing:
        logger.warning("Missing label IDs: %s", missing)
    return kinetics_id_to_classname


def score_frame_quality(frame_rgb):
    """Small heuristic to avoid especially blurry or low-information frames."""
    import cv2

    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    contrast = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    exposure_penalty = abs(brightness - 128.0) / 128.0
    return (sharpness * 0.65) + (contrast * 0.35) - (exposure_penalty * 20.0)


def select_quality_frames(frames, target_frame_count):
    """Select one stable frame per time bin while preserving order."""
    if len(frames) <= target_frame_count:
        return frames

    bin_edges = np.linspace(0, len(frames), target_frame_count + 1).astype(int)
    selected = []
    for start, end in zip(bin_edges[:-1], bin_edges[1:]):
        chunk = frames[start : max(start + 1, end)]
        selected.append(max(chunk, key=score_frame_quality))
    return selected


def maybe_normalize_video_container(video_path, target_fps=30, clip_duration=TARGET_CLIP_SECONDS):
    """Rewrite awkward containers into a short MP4 clip for stabler inference."""
    import cv2

    source_path = Path(video_path)
    if not NORMALIZE_CONTAINER_FOR_INFERENCE or source_path.suffix.lower() not in NORMALIZE_EXTENSIONS:
        return video_path, {"normalized_container": False}

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        return video_path, {"normalized_container": False, "normalize_error": "opencv_open_failed"}

    fps = cap.get(cv2.CAP_PROP_FPS) or float(target_fps)
    max_read_frames = int(max(fps, target_fps) * min(MAX_SOURCE_SECONDS, max(clip_duration, 1.0)))
    frames = []
    while len(frames) < max_read_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if len(frames) < 4:
        return video_path, {"normalized_container": False, "normalize_error": "too_few_frames"}

    desired_frames = min(len(frames), max(int(target_fps * clip_duration), 48))
    if len(frames) > desired_frames:
        start_index = max((len(frames) - desired_frames) // 2, 0)
        frames = frames[start_index : start_index + desired_frames]

    NORMALIZED_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    temp_handle = tempfile.NamedTemporaryFile(
        dir=NORMALIZED_VIDEO_DIR,
        prefix=f"{source_path.stem}_",
        suffix=".mp4",
        delete=False,
    )
    temp_handle.close()

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        temp_handle.name,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(target_fps),
        (width, height),
    )
    for frame in frames:
        writer.write(frame)
    writer.release()

    return temp_handle.name, {
        "normalized_container": True,
        "normalized_path": temp_handle.name,
        "normalized_frame_count": len(frames),
        "normalized_target_fps": target_fps,
    }


def load_video_with_opencv(video_path, num_frames=32, target_fps=30):
    """
    Loads video frames using OpenCV and returns a tensor [C, T, H, W].
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"OpenCV cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(
        "Video info: %sx%s, %.1ffps, %s frames, duration=%.1fs",
        width,
        height,
        fps,
        total_frames,
        (total_frames / fps) if total_frames else 0.0,
    )

    if 0 < total_frames < 4:
        cap.release()
        raise ValueError(f"Video too short: only {total_frames} frames")

    max_read_frames = (
        min(total_frames, int(fps * MAX_SOURCE_SECONDS))
        if total_frames > 0
        else int(target_fps * MAX_SOURCE_SECONDS)
    )
    frames = []
    for _ in range(max_read_frames):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if len(frames) < 4:
        raise ValueError(f"Could only read {len(frames)} frames from video")

    logger.info("Read %s frames from video", len(frames))

    desired_clip_frames = min(len(frames), max(int(target_fps * TARGET_CLIP_SECONDS), num_frames * 2))
    trim_start = 0
    trimmed_frames = frames
    if len(frames) > desired_clip_frames:
        trim_start = max((len(frames) - desired_clip_frames) // 2, 0)
        trimmed_frames = frames[trim_start : trim_start + desired_clip_frames]

    selected_frame_count = min(len(trimmed_frames), max(num_frames * 2, 48))
    processed_frames = select_quality_frames(trimmed_frames, selected_frame_count)

    video_tensor = torch.from_numpy(np.array(processed_frames)).float()
    video_tensor = video_tensor.permute(3, 0, 1, 2)

    preprocessing = {
        "decoder": "opencv",
        "sampling_strategy": "center_trim+quality_bins",
        "source_fps": round(float(fps), 2),
        "source_frames_read": len(frames),
        "center_trim_start": trim_start,
        "trimmed_frame_count": len(trimmed_frames),
        "selected_frame_count": len(processed_frames),
        "target_clip_seconds": TARGET_CLIP_SECONDS,
    }
    return video_tensor, preprocessing


def load_model(device):
    """Loads the pre-trained SlowFast ResNet-50 model."""
    logger.info("Loading SlowFast model...")
    TORCH_HUB_DIR.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(TORCH_HUB_DIR))
    logger.info("Torch Hub cache directory: %s", TORCH_HUB_DIR)
    model = torch.hub.load("facebookresearch/pytorchvideo", "slowfast_r50", pretrained=True)
    model = model.eval().to(device)
    logger.info("SlowFast model loaded successfully.")
    return model


class SlowFastInferenceEngine:
    """
    Encapsulates the entire SlowFast inference pipeline and Training Lab feature extraction.
    """

    def __init__(self):
        self.model_name = MODEL_NAME
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Initializing SlowFast Inference Engine on device: %s", self.device)

        self.model = load_model(self.device)
        self.transform = get_transform()
        self.labels = download_kinetics_labels()

        logger.info("Engine ready. Loaded %s Kinetics-400 labels.", len(self.labels))
        sample_labels = [self.labels.get(i, "MISSING") for i in range(5)]
        logger.info("Label check (IDs 0-4): %s", sample_labels)

    def _load_video_frames(self, video_path: str):
        """
        Attempts to load video using multiple methods for maximum compatibility.
        Returns a dict with a 'video' key containing the raw video tensor.
        """
        try:
            logger.info("Loading video with OpenCV...")
            video_tensor, preprocessing = load_video_with_opencv(video_path)
            logger.info(
                "OpenCV loaded video tensor: shape=%s, range=[%.1f, %.1f]",
                tuple(video_tensor.shape),
                video_tensor.min(),
                video_tensor.max(),
            )
            return {"video": video_tensor}, preprocessing
        except Exception as exc:
            logger.warning("OpenCV loading failed: %s. Trying pytorchvideo...", exc)

        try:
            logger.info("Loading video with EncodedVideo (PyAV)...")
            video = EncodedVideo.from_path(video_path)
            num_frames = 32
            sampling_rate = 2
            fps = 30
            clip_duration = (num_frames * sampling_rate) / fps
            video_data = video.get_clip(start_sec=0, end_sec=clip_duration)
            if video_data is None or video_data.get("video") is None:
                raise ValueError("get_clip returned None - video may be corrupted or too short")
            logger.info(
                "EncodedVideo loaded: shape=%s, range=[%.1f, %.1f]",
                tuple(video_data["video"].shape),
                video_data["video"].min(),
                video_data["video"].max(),
            )
            return video_data, {
                "decoder": "pyav",
                "sampling_strategy": "encoded_video_clip",
                "target_clip_seconds": TARGET_CLIP_SECONDS,
            }
        except Exception as exc:
            logger.error("EncodedVideo loading also failed: %s", exc)
            raise ValueError(f"Could not load video with any decoder: {str(exc)}") from exc

    def _prepare_model_inputs(self, video_path: str):
        """Load a clip and convert it into SlowFast pathway tensors."""
        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")

        file_size = os.path.getsize(video_path)
        logger.info("File size: %.2f MB", file_size / 1e6)
        if file_size < 1000:
            raise ValueError(f"Video file too small ({file_size} bytes) - likely corrupted")

        normalized_path = None
        normalized_meta = {"normalized_container": False}
        analysis_path = video_path

        try:
            analysis_path, normalized_meta = maybe_normalize_video_container(video_path)
            if analysis_path != video_path:
                normalized_path = analysis_path
                logger.info("Using normalized MP4 for inference: %s", analysis_path)
        except Exception as exc:
            logger.warning("Container normalization skipped: %s", exc)
            analysis_path = video_path

        try:
            video_data, preprocessing = self._load_video_frames(analysis_path)
            if video_data is None or video_data.get("video") is None:
                raise ValueError("Failed to extract video data - file may be too short or corrupted.")

            raw_video = video_data["video"]
            raw_std = float(raw_video.std().item())
            raw_min = float(raw_video.min().item())
            raw_max = float(raw_video.max().item())
            logger.info(
                "Raw video tensor: shape=%s, dtype=%s, range=[%.1f, %.1f], std=%.2f",
                tuple(raw_video.shape),
                raw_video.dtype,
                raw_min,
                raw_max,
                raw_std,
            )

            if raw_std < 3.0:
                raise ValueError("Video frames appear blank or nearly uniform; no usable visual content found")

            video_data = self.transform(video_data)
            inputs = [tensor.to(self.device)[None, ...] for tensor in video_data["video"]]
            logger.info("After transforms: slow=%s, fast=%s", inputs[0].shape, inputs[1].shape)
            return inputs, {**preprocessing, **normalized_meta}, normalized_path
        except Exception:
            if normalized_path and os.path.exists(normalized_path):
                os.unlink(normalized_path)
            raise

    def extract_embedding(self, video_path: str) -> dict:
        """Return a pooled SlowFast backbone embedding for a single clip."""
        logger.info("Extracting embedding from: %s", video_path)
        normalized_path = None
        try:
            inputs, preprocessing, normalized_path = self._prepare_model_inputs(video_path)
            with torch.no_grad():
                features = inputs
                for block in self.model.blocks[:6]:
                    features = block(features)

                if isinstance(features, (list, tuple)):
                    pooled = [F.adaptive_avg_pool3d(part, output_size=1).flatten(1) for part in features]
                    embedding = torch.cat(pooled, dim=1)
                elif isinstance(features, torch.Tensor):
                    embedding = (
                        F.adaptive_avg_pool3d(features, output_size=1).flatten(1)
                        if features.ndim == 5
                        else features.flatten(1)
                    )
                else:
                    raise ValueError("Unexpected feature structure returned by SlowFast.")

            embedding = embedding.squeeze(0).detach().cpu()
            return {
                "success": True,
                "embedding": embedding,
                "embedding_dim": int(embedding.shape[0]),
                "device": self.device,
                "preprocessing": preprocessing,
                "error": None,
            }
        except Exception as exc:
            logger.error("Embedding extraction failed: %s", exc)
            return {
                "success": False,
                "embedding": None,
                "embedding_dim": 0,
                "device": self.device,
                "preprocessing": {},
                "error": str(exc),
            }
        finally:
            if normalized_path and os.path.exists(normalized_path):
                os.unlink(normalized_path)

    def predict(self, video_path: str, top_k: int = 5) -> dict:
        """Run clip-level inference and return top-k Kinetics predictions."""
        logger.info("Running inference on: %s", video_path)
        try:
            inputs, preprocessing_payload, normalized_path = self._prepare_model_inputs(video_path)
        except Exception as exc:
            return self._error_result(f"Failed to prepare video: {str(exc)}")

        logger.info("Running model forward pass...")
        try:
            with torch.no_grad():
                preds = self.model(inputs)
        except Exception as exc:
            logger.error("Model forward pass failed: %s", exc)
            if normalized_path and os.path.exists(normalized_path):
                os.unlink(normalized_path)
            return self._error_result(f"Model inference failed: {str(exc)}")

        logger.info("Raw model output shape: %s", tuple(preds.shape))
        preds = torch.nn.Softmax(dim=1)(preds)
        topk_result = preds.topk(k=min(top_k, preds.shape[1]))
        top_indices = topk_result.indices[0]
        top_scores = topk_result.values[0]

        predictions = []
        for i in range(min(top_k, len(top_indices))):
            idx = int(top_indices[i].item())
            label = self.labels.get(idx, f"class_{idx}")
            conf = float(top_scores[i].item())
            predictions.append({"label": label, "confidence": round(conf * 100, 2)})
            logger.info("  #%s: [%s] %s = %.2f%%", i + 1, idx, label, conf * 100)

        if normalized_path and os.path.exists(normalized_path):
            os.unlink(normalized_path)

        if not predictions:
            return self._error_result("Model produced no predictions")

        result = {
            "success": True,
            "predictions": predictions,
            "top_prediction": predictions[0]["label"],
            "top_confidence": predictions[0]["confidence"],
            "device": self.device,
            "preprocessing": preprocessing_payload,
            "error": None,
        }
        logger.info(
            "Inference complete. Top prediction: %s (%.2f%%)",
            result["top_prediction"],
            result["top_confidence"],
        )
        return result

    def _error_result(self, error_msg: str) -> dict:
        """Helper to create a standardized error result."""
        logger.error("Inference error: %s", error_msg)
        return {
            "success": False,
            "predictions": [],
            "top_prediction": None,
            "top_confidence": 0,
            "device": self.device,
            "error": error_msg,
        }


def main():
    parser = argparse.ArgumentParser(description="SlowFast Video Inference Script")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video file")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top predictions to return")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video file '{args.video}' not found.")
        return

    engine = SlowFastInferenceEngine()
    result = engine.predict(args.video, top_k=args.top_k)

    if result["success"]:
        print(f"\n{'=' * 40}")
        print(f"  TOP {args.top_k} PREDICTIONS ({result['device'].upper()})")
        print(f"{'=' * 40}")
        for i, pred in enumerate(result["predictions"]):
            bar_len = int(pred["confidence"] / 2)
            bar = "#" * bar_len + "." * (50 - bar_len)
            print(f"  {i + 1}. {pred['label']:30s} {bar} {pred['confidence']:.1f}%")
    else:
        print(f"Error: {result['error']}")


if __name__ == "__main__":
    main()
