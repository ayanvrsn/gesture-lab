# Gesture Lab

Source code for a local Flask gesture-recognition demo using MobileNetV3-Small, training-only augmentation, Grad-CAM, environment metrics, latency and a controlled stress panel. Model weights and image data are not included in this repository.

**Status:** functional local inference package; the assignment's mandatory Step-Video-T2V / ArtAug experiment is not completed. See `sota/README.txt` for the experiment status and continuation notes. No claim of SOTA superiority is made.

## Quick start
Python 3.12, from this folder:

```sh
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: use .venv\Scripts\python.exe instead of python below
python -m pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:8000. Upload an image for automatic prediction, or click **Start camera** for continuous prediction (target maximum 6 updates/s). **Explain frame** pauses the camera and computes Grad-CAM. Actual update rate depends on the device. Model weights and data are supplied separately in Gesture_Assignment2_Package.zip. Copy its data/ and results/ folders into this repository before running. Uploads are processed in memory; the server binds to loopback.

## What is included

- `app.py`, `inference.py`, `templates/index.html`: local web app.
- `prepare.py`, `source_assignments.csv`: input audit, source split assignments and augmentation. Running preparation generates `data/manifest.csv`.
- `train.py`, `predict.py`: training and command-line inference.
- `verify.py`, `test_app.py`, `evaluate_demo.py`: data, API and scene/stress checks.
- `results/`: recorded metrics, curves, predictions and archived A1 evidence.
- `sota/`: continuation notes and an adapter for externally generated reviewed clips; no generated SOTA output included.

The source archive does not contain the dataset, model checkpoints, report PDF, walkthrough video or defense documents. To run the demo, obtain the companion package separately and place the checkpoint at `results/mobilenet_v3_small_gestures.pth`. Data-dependent checks also require the companion `data/` folder.

## Results

| Method | Kitchen test accuracy | Macro-F1 |
|---|---:|---:|
| A1 landmarks + SVM (archived) | 29/29, 100% | 1.000 |
| MobileNet, originals repeated | 17/29, 58.6% | 0.575 |
| MobileNet, augmented | 15/29, 51.7% | 0.528 |

150 source photos, 146 retained: 88 train + 29 validation + 29 test. Training expands to 5,400 files using 5,312 classical variants. Same A1 evaluation exclusions and scene holdouts. Participant-disjoint separation and public redistribution consent are unverified. The A1 score is conditional on detector success. No unknown-gesture class is implemented.

## Reproduction

```sh
python verify.py
python test_app.py
python evaluate_demo.py
python train.py
```

Training and evaluation overwrite their result files; back up first. Training downloads ImageNet initialization unless `--weights /path/to/mobilenet_v3_small-047dcff4.pth` is supplied. Frozen ImageNet backbone, trainable head, AdamW lr=0.0003, batch=96, 6 epochs, seed=17, 5,400 exposures/epoch for both conditions. Select by validation macro-F1, then validation loss. All augmentations are restricted to train. Raw image hashes and parent assignments are retained.

The demo runs on CPU. The supplied archive reports that automated API tests and data checks passed in Linux/Python 3.12; those checks require the separately supplied weights and data. Camera and full browser layout need a user-device check; a separate cloud browser could not reach this local server. No public hosting or Moodle submission is included.

## Repository

This repository contains the source-only archive imported in logical commits; the commit sequence describes the import, not the original development timeline. `.gitignore` excludes data, model weights and video; provide them as the private companion archive. Do not push participant images publicly without consent.
