# NeuralVision AI

NeuralVision AI is a local video understanding application built with FastAPI, React, and PyTorchVideo SlowFast. It combines clip-based action recognition, explainability, evaluation, review workflows, and a lightweight Training Lab in one project.

## Overview

The project is designed as an end-to-end local AI workflow:

- upload a video or record a short webcam clip
- run action recognition with a pretrained SlowFast R50 model
- return confidence-aware predictions with an `uncertain` fallback
- generate optional Gemini-based explanations and chat responses
- store analysis history and buffer samples for future review
- evaluate the active model on labeled manifests
- train and inspect a small pilot model in Training Lab

## Core Features

### Analysis Workspace
- video upload and webcam capture
- top-k action predictions
- confidence thresholding and uncertainty handling
- prediction explanations
- recent analysis history with thumbnails and export

### Chat Workspace
- chat-style interface inside the app
- conversation history, rename, delete, and export
- backend-side Gemini integration
- optional video context attached to chat prompts

### Evaluation and Review
- run evaluation from a labeled JSONL manifest
- inspect latest evaluation summary and export reports
- review buffered clips and write reviewed labels for future training

### Training Lab
- separate workspace from the main analysis flow
- scans a local five-class pilot dataset: `clap`, `wave`, `punch`, `talk`, `walk`
- launches a preset training run
- stores run metadata, logs, metrics, and checkpoints
- supports manual promotion of a completed pilot model

## Tech Stack

- **Backend:** FastAPI, Uvicorn, PyTorch, PyTorchVideo, OpenCV
- **Frontend:** React, Vite, plain CSS
- **Modeling:** SlowFast R50 for general action recognition, lightweight pilot classifier for Training Lab
- **Storage:** local JSON/JSONL files and local filesystem directories

## Repository Layout

```text
.
|-- main.py                         # FastAPI application and API routes
|-- inference.py                    # SlowFast inference and feature extraction
|-- training_lab.py                 # Pilot training manager and checkpoint flow
|-- evaluation.py                   # Evaluation pipeline
|-- data_buffer.py                  # Buffering and future-training data handling
|-- explainability.py               # Gemini explanation integration
|-- chat_service.py                 # Chat session persistence and helpers
|-- frontend/                       # React client
|-- configs/training/               # Training config scaffolds
|-- datasets/reviewed/              # Reviewed-dataset scaffold
|-- models/                         # Model registry metadata
`-- data/                           # Local runtime data, caches, exports, runs
```

## Quick Start

### 1. Verify the Python environment

```powershell
.\venv\Scripts\python.exe check_env.py
```

### 2. Start the backend

```powershell
.\venv\Scripts\python.exe main.py
```

### 3. Start the frontend

```powershell
cd frontend
npm run dev
```

Open the Vite URL shown in the terminal, usually `http://127.0.0.1:5173`.

## Configuration

Runtime settings are controlled through environment variables. Use [.env.example](.env.example) as the reference list.

Typical local startup:

```powershell
$env:GEMINI_API_KEY="your_key_here"
$env:CONFIDENCE_THRESHOLD="25"
$env:MAX_UPLOAD_MB="200"
.\venv\Scripts\python.exe main.py
```

Useful inference-related settings:

- `TARGET_CLIP_SECONDS`
- `MAX_SOURCE_SECONDS`
- `NORMALIZE_CONTAINER_FOR_INFERENCE`
- `TOP1_MIN_CONFIDENCE`
- `TOP1_TOP2_GAP_THRESHOLD`

## Typical Workflows

### Run Video Analysis
1. Start the backend and frontend.
2. Open the Analysis workspace.
3. Upload a clip or record a short webcam video.
4. Inspect the prediction, confidence, and explanation.

### Run Evaluation
1. Create `data/evaluation/manifest.jsonl`.
2. Add one JSON object per line, for example:

```json
{"path":"data/evaluation/videos/sample_punch.mp4","label":"punch","notes":"optional"}
```

3. Run evaluation from the UI or with:

```powershell
.\venv\Scripts\python.exe evaluate_model.py
```

### Use Training Lab
1. Place clips into:
   - `data/training_lab/basic_actions_dataset/clap`
   - `data/training_lab/basic_actions_dataset/wave`
   - `data/training_lab/basic_actions_dataset/punch`
   - `data/training_lab/basic_actions_dataset/talk`
   - `data/training_lab/basic_actions_dataset/walk`
2. Make sure each class has at least 5 clips.
3. Open **Training Lab** in the UI and start a preset run.
4. Review logs, metrics, and checkpoints.
5. Promote a successful run only when you want it active in Analysis.

## API Summary

- `GET /health`
- `POST /upload-video`
- `GET /explanation`
- `GET /analysis/history`
- `GET /evaluation/latest`
- `POST /evaluation/run`
- `GET /models/current`
- `POST /models/current`
- `GET /training-lab/overview`
- `POST /training-runs/start`

## Current Scope

This project is strongest as a **clip-level action recognition system**. It predicts the dominant action in a short video clip. It is not a multi-person action detection system with bounding boxes and per-person labels.

The default production path uses a pretrained Kinetics-400 SlowFast model. Training Lab demonstrates how the project can evolve toward more specialized models over time.

## Demo Guidance

For the most reliable live demo, use short clips with:

- one clear subject
- one obvious action
- stable framing
- good lighting
- minimal background clutter

Actions such as clapping, instrument playing, push-ups, boxing motion, and sports actions generally present better than noisy multi-person scenes.

## Notes on Accuracy

- The default model is broad, not domain-specific.
- Low-confidence outputs are intentionally reported as `uncertain`.
- The pilot model in Training Lab is useful for demonstrating iterative improvement, not as a guaranteed replacement for the pretrained model.

## Project Status

NeuralVision AI is currently a polished local prototype with:

- a working backend and frontend
- a functional analysis workflow
- explainability and chat integration
- evaluation and review tooling
- a separate training workspace for pilot model development

The biggest future improvement remains the same: more reviewed data and better fine-tuned models.
