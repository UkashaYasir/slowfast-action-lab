import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files


REPO_ID = "divm/hmdb51"
REPO_TYPE = "dataset"
TARGET_CLASSES = ("clap", "wave", "punch", "talk", "walk")
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".webm", ".mkv", ".wmv", ".flv", ".m4v"}


def score_candidate(split_name: str, filename: str) -> int:
    stem_parts = Path(filename).stem.split("_")
    tags = stem_parts[-6:-1] if len(stem_parts) >= 6 else []
    score = 0

    score += 30 if split_name == "train" else 18
    if "np1" in tags:
        score += 35
    elif "np2" in tags:
        score += 10
    else:
        score -= 5

    if "fr" in tags:
        score += 18
    if "nm" in tags:
        score += 14
    if "ri" in tags or "le" in tags:
        score += 8
    if "ba" in tags:
        score -= 4

    if "goo" in tags:
        score += 20
    elif "med" in tags:
        score += 10
    elif "bad" in tags:
        score -= 12

    return score


def choose_files(files: list[str], label: str, limit: int) -> list[str]:
    ranked = []
    for path in files:
        parts = path.split("/", 1)
        if len(parts) != 2:
            continue
        split_name, filename = parts
        if not filename.startswith(f"{label}_"):
            continue
        if Path(filename).suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        ranked.append((score_candidate(split_name, filename), path))

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [path for _, path in ranked[:limit]]


def main():
    parser = argparse.ArgumentParser(description="Populate Training Lab dataset folders from HMDB51.")
    parser.add_argument("--per-class", type=int, default=8, help="How many clips to download per class.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/training_lab/basic_actions_dataset",
        help="Base directory for the Training Lab dataset.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Clear existing class folders before downloading the curated subset.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_files = list_repo_files(REPO_ID, repo_type=REPO_TYPE)
    print(f"Found {len(repo_files)} files in {REPO_ID}.")

    for label in TARGET_CLASSES:
        label_dir = output_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        if args.replace:
            for existing in label_dir.iterdir():
                if existing.is_file():
                    existing.unlink()

        candidates = choose_files(repo_files, label, args.per_class)
        if len(candidates) < args.per_class:
            print(f"[warn] Only found {len(candidates)} candidates for {label}.")

        downloaded = 0
        for repo_path in candidates:
            filename = Path(repo_path).name
            dest_path = label_dir / filename
            if dest_path.exists():
                downloaded += 1
                continue

            local_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=repo_path,
                repo_type=REPO_TYPE,
            )
            shutil.copy2(local_path, dest_path)
            downloaded += 1
            print(f"[ok] {label}: {filename}")

        print(f"{label}: ready with {downloaded} clip(s) in {label_dir}")


if __name__ == "__main__":
    main()
