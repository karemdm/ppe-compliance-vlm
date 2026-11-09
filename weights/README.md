# Weights

All three YOLO weights used by the project are stored in this folder.

| File | Used for | Source | License | Size | SHA-256 |
|------|----------|--------|---------|------|---------|
| `yolov8n.pt` | Person detection + ByteTrack tracking (COCO class 0) | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) | AGPL-3.0 | 6.5 MB (6,549,796 bytes) | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` |
| `safetyvision_v2_yolov8s.pt` | PPE model `safetyvision` (default), YOLOv8s, 13 classes | [ayushgupta7777/safetyvision-yolov8](https://huggingface.co/ayushgupta7777/safetyvision-yolov8), file `v2/best.pt` | AGPL-3.0 | 22 MB (22,547,434 bytes) | `7863be4700dcf831579d610bb3fe3668fb29fb22ab17ca027b55e94b88bfff7a` |
| `chd_yolo26l.pt` | PPE model `chd`, YOLO26-L, 11 classes | [yihong1120/Construction-Hazard-Detection](https://huggingface.co/yihong1120/Construction-Hazard-Detection), file `models/yolo26/pt/yolo26l.pt` (code: [GitHub](https://github.com/yihong1120/Construction-Hazard-Detection)) | AGPL-3.0 | 51 MB (52,974,949 bytes) | `398c424c1edbc0d288f88ecb9b38ac30d941707142f74db8b622b1b82f8d10d3` |

Check a file with `sha256sum weights/<file>.pt` (Linux/macOS) or
`Get-FileHash weights\<file>.pt -Algorithm SHA256` (PowerShell).

## Person model

`weights/yolov8n.pt` is the default of `--person-weights`. If the file is
missing, the file name `yolov8n.pt` is passed to Ultralytics, which downloads
the official weights automatically.

## PPE models

The PPE model is chosen with `--ppe-model {safetyvision,chd}` or in the
Streamlit sidebar. The files are looked up relative to the current directory
and then to the repository root; they are **not** downloaded automatically,
so a missing file raises an error.

| Key | Class names (as stored in the weights) | Inference `imgsz` |
|-----|-----------------------------------------|-------------------|
| `safetyvision` | Fall-Detected, Gloves, Goggles, Hardhat, Mask, NO-Gloves, NO-Goggles, NO-Hardhat, NO-Mask, NO-Safety Vest, No_Harness, Person, Safety Vest | 896 (recommended on the model card) |
| `chd` | Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person, Safety Cone, Safety Vest, machinery, utility pole, vehicle | 640 (used on the model card) |

The mapping of these classes to PPE items (worn / missing pairs) is defined
in `src/ppe_compliance/models.py`. When a model is loaded, its class names
are compared with the registry and a warning is logged if they differ.

> **License note:** both PPE models are published under AGPL-3.0 by their
> authors. Review the license obligations on the model cards before using
> them in a networked or commercial deployment.

## VLM

The report text uses `HuggingFaceM4/idefics2-8b` (Apache-2.0), downloaded at
runtime by `transformers` and not stored here.
