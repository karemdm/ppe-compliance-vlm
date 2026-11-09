# PPE detection: model, dataset and test-video research

> Research notes prepared with AI assistance (Claude) on 2026-11-09. Every number below is copied from the cited source and has **not** been reproduced or validated by this project. Treat it as a shortlist to evaluate, not as verified results.

## 1. Summary / recommendation

The current model, [`Hexmon/vyra-yolo-ppe-detection`](https://huggingface.co/Hexmon/vyra-yolo-ppe-detection) (YOLOv8m, 14 classes, trained on Roboflow "PPE Combined Model v4", CC BY 4.0), publishes **no evaluation metrics** at all, so its quality cannot be checked from the card ([source](https://huggingface.co/Hexmon/vyra-yolo-ppe-detection)).

**Top 1: [`ayushgupta7777/safetyvision-yolov8`](https://huggingface.co/ayushgupta7777/safetyvision-yolov8), file `v2/best.pt`**
- It is a near drop-in replacement. It covers almost the same taxonomy as the current model (`Fall-Detected`, `NO-Hardhat`, `NO-Safety Vest`, `NO-Gloves`, `NO-Goggles`, `NO-Mask`, plus the positive classes), so the app's infraction set maps 1:1.
- It is the only candidate found that publishes **test-split** metrics, per class: mAP50 0.766 and mAP50-95 0.487 on 4,026 test images at imgsz 896 ([source](https://huggingface.co/ayushgupta7777/safetyvision-yolov8)).
- It was trained on about 80k deduplicated images from 5 Roboflow Universe datasets, one of which is a PPE-combined dataset similar to the one the current model used ([source](https://huggingface.co/ayushgupta7777/safetyvision-yolov8)).
- Caveats:
  - `NO-Safety Vest` is weak (test mAP50 0.386), and the card itself advises against using it as the only vest signal ([source](https://huggingface.co/ayushgupta7777/safetyvision-yolov8)).
  - The weights are AGPL-3.0.
  - Dataset licenses are not stated on the card.
  - It was trained at 896 px. Run inference at `imgsz=896` to match the reported numbers.

**Top 2: [`yihong1120/Construction-Hazard-Detection`](https://huggingface.co/yihong1120/Construction-Hazard-Detection), YOLO26 `.pt` (for example `models/yolo26/pt/yolo26s.pt` or `yolo26l.pt`)**
- It is actively maintained, uses current Ultralytics YOLO26, and comes in 5 sizes. It publishes validation mAP50-95 per size, for example YOLO26s at 0.5996 and YOLO26x at 0.6673 on 6,664 validation images ([source](https://huggingface.co/yihong1120/Construction-Hazard-Detection)).
- It has explicit `NO-Hardhat`, `NO-Safety Vest` and `NO-Mask` classes ([source](https://huggingface.co/yihong1120/Construction-Hazard-Detection)).
- It has **no gloves, goggles or fall classes**. Use it as a hardhat/vest specialist, or alongside the top-1 model.
- The license is AGPL-3.0 ([source](https://huggingface.co/yihong1120/Construction-Hazard-Detection)).
- Its metrics are on its own validation set and cannot be compared directly with SafetyVision's test set.

**Suggested next step:** run both against a small, hand-labelled clip set from the target domain before switching. A minimal check is `model.val()` on a few hundred labelled frames. Do not rely on the card numbers alone.

### How to plug in (app loads via `hf_hub_download(repo_id, filename)` and maps lowercase class names)

**SafetyVision v2:** `hf_hub_download("ayushgupta7777/safetyvision-yolov8", "v2/best.pt")`. Class names are taken verbatim from the card ([source](https://huggingface.co/ayushgupta7777/safetyvision-yolov8)).

| id | model class (exact) | lowercase key | app set |
|---|---|---|---|
| 0 | `Fall-Detected` | `fall-detected` | infraction (fall-detected) |
| 1 | `Gloves` | `gloves` | positive (gloves) |
| 2 | `Goggles` | `goggles` | positive (goggles) |
| 3 | `Hardhat` | `hardhat` | positive (hardhat/helmet) |
| 4 | `Mask` | `mask` | positive (mask) |
| 5 | `NO-Gloves` | `no-gloves` | infraction (no-gloves) |
| 6 | `NO-Goggles` | `no-goggles` | infraction (no-goggles) |
| 7 | `NO-Hardhat` | `no-hardhat` | infraction (no-hardhat) |
| 8 | `NO-Mask` | `no-mask` | infraction (no-mask) |
| 9 | `NO-Safety Vest` | `no-safety vest` | infraction (no-safety vest) |
| 10 | `No_Harness` | `no_harness` | **new**: not in the app; add it as an infraction or ignore it |
| 11 | `Person` | `person` | neither (useful for per-person association) |
| 12 | `Safety Vest` | `safety vest` | positive (safety vest/vest) |

There is no `boots` class. The current model's `Ladder` and `Safety Cone` classes disappear, which is harmless.

**Construction-Hazard-Detection (YOLO26):** for example `hf_hub_download("yihong1120/Construction-Hazard-Detection", "models/yolo26/pt/yolo26s.pt")`. The path layout comes from the [HF card](https://huggingface.co/yihong1120/Construction-Hazard-Detection). The GitHub README mentions an older `models/pt/best_<model_key>.pt` layout ([source](https://github.com/yihong1120/Construction-Hazard-Detection)), so confirm the path in "Files and versions" before hard-coding it.

| id | model class (exact) | lowercase key | app set |
|---|---|---|---|
| 0 | `Hardhat` | `hardhat` | positive |
| 1 | `Mask` | `mask` | positive |
| 2 | `NO-Hardhat` | `no-hardhat` | infraction |
| 3 | `NO-Mask` | `no-mask` | infraction |
| 4 | `NO-Safety Vest` | `no-safety vest` | infraction |
| 5 | `Person` | `person` | neither |
| 6 | `Safety Cone` | `safety cone` | neither |
| 7 | `Safety Vest` | `safety vest` | positive |
| 8 | `machinery` | `machinery` | neither |
| 9 | `utility pole` | `utility pole` | neither |
| 10 | `vehicle` | `vehicle` | neither |

The HF card writes ids 8–10 in lowercase; the duplicate README writes them capitalised ([source](https://huggingface.co/architchitte/Construction-Hazard-Detection/blob/main/README.md)). Lowercase mapping makes this irrelevant. Read `model.names` at runtime instead of hard-coding ids.

## 2. Candidate models

All metrics are self-reported by the authors. "Split" says what the number refers to.

| # | Model (link) | Arch | Ultralytics `.pt`? | Classes (missing-PPE classes?) | Training data / license | Weights license | Published metrics | Last update | Red flags |
|---|---|---|---|---|---|---|---|---|---|
| 0 (current) | [Hexmon/vyra-yolo-ppe-detection](https://huggingface.co/Hexmon/vyra-yolo-ppe-detection) | YOLOv8m | Yes, `best.pt` | 14: Fall-Detected, Gloves, Goggles, Hardhat, Ladder, Mask, NO-Gloves, NO-Goggles, NO-Hardhat, NO-Mask, NO-Safety Vest, Person, Safety Cone, Safety Vest (yes) | Roboflow "PPE Combined Model v4", CC BY 4.0 (per card) | CC BY 4.0 (card) | **None published** | Not shown | No metrics; card license CC BY 4.0 even though Ultralytics is AGPL-3.0 |
| 1 | [ayushgupta7777/safetyvision-yolov8](https://huggingface.co/ayushgupta7777/safetyvision-yolov8) | YOLOv8s (v2) | Yes, `v2/best.pt` (~22.5 MB); v1 `best.pt` at root | 13: Fall-Detected, Gloves, Goggles, Hardhat, Mask, NO-Gloves, NO-Goggles, NO-Hardhat, NO-Mask, NO-Safety Vest, No_Harness, Person, Safety Vest (yes) | 5 Roboflow Universe datasets, ~80,304 images after dedup; **dataset licenses not stated** | AGPL-3.0 | Test (4,026 img, imgsz 896): mAP50 0.766, mAP50-95 0.487. Val: 0.787 / 0.504. Per-class published; NO-Safety Vest 0.386 | Not shown; W&B run id suggests training 2026-05-19 (unverified) | Weak NO-Safety Vest; Western-site bias (stated); AGPL |
| 2 | [yihong1120/Construction-Hazard-Detection](https://huggingface.co/yihong1120/Construction-Hazard-Detection) ([GitHub](https://github.com/yihong1120/Construction-Hazard-Detection)) | YOLO26 n/s/m/l/x (YOLO11 legacy) | Yes, `models/yolo26/pt/*.pt` (also ONNX, TensorRT) | 11: Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person, Safety Cone, Safety Vest, machinery, utility pole, vehicle (yes, but no gloves/goggles/fall) | Roboflow "construction-hazard-detection" project; license not stated on GitHub page | AGPL-3.0 | **Val** mAP50-95 (6,664 img, imgsz 640, FP32): n 0.4765, s 0.5996, m 0.5891, l 0.6482, x 0.6673. No mAP50 given | No date on HF; GitHub shows 972 commits, Ultralytics 8.4.x | Validation only; covers hardhat/vest/mask only |
| 3 | [killuminati1/construction-ppe-yolov8](https://huggingface.co/killuminati1/construction-ppe-yolov8) | YOLOv8 (size not stated) | Probably (`best.pt` in snippet, unconfirmed) | 19 incl. Boots, Glass, Glove, Hard_hat, Mask, Vest, No-Boots, No-Glass, No-Glove, No-Helmet, No-Mask, No-Vest, No-Ear-Protection (yes, incl. boots) | ~13k images, "primarily Roboflow" + one GitHub set; not named | Apache-2.0 (card), base AGPL | mAP50 ~0.76, mAP50-95 ~0.43, **split not stated** | Not shown | Stopped at 60 epochs while loss still falling; datasets unnamed; Apache label on AGPL-derived weights |
| 4 | [Hansung-Cho/yolov8-ppe-detection](https://huggingface.co/Hansung-Cho/yolov8-ppe-detection) | YOLOv8n | Yes, `best.pt` | "~10" classes, examples: Person, Hardhat, No-Hardhat, Safety Vest, No-Safety Vest, Mask, No-Mask (yes, full list not published) | "Public PPE datasets (Kaggle etc.)", unnamed | MIT (card) | Val: mAP50 0.744, mAP50-95 0.436 | Citation says 2025 | Class list incomplete; nano model; MIT label on AGPL-derived weights |
| 5 | [Nexus3/yolov8n-ppe-detection-6classes](https://huggingface.co/Nexus3/yolov8n-ppe-detection-6classes) | YOLOv8n | Yes, `best.pt` | 6: Gloves, Vest, goggles, helmet, mask, safety_shoe (**no** negative classes) | "Custom PPE dataset (Roboflow format)", license not stated | MIT (card) | mAP50 ~0.81, mAP50-95 ~0.53, split not stated | Not shown; duplicated from Tanishjain9/... | Positive-only: "missing" must be inferred per person |
| 6 | [melihuzunoglu/ppe-detection](https://huggingface.co/melihuzunoglu/ppe-detection) | YOLO11 | Yes, `best.pt` (per sample code) | 4: Human, Helmet, No-Helmet, Vest (No-Vest inferred, not trained) | Unnamed Roboflow-curated set | AGPL-3.0 | **None published** | Not shown | No metrics; very small taxonomy |
| 7 | [SH17 weights (ahmadmughees/SH17dataset)](https://github.com/ahmadmughees/SH17dataset) | YOLOv8 n–x, YOLOv9 t–e, YOLOv10 n–x | Yes, GitHub release `releases/download/v1/<file>.pt` (e.g. `yolo9e.pt`) | 17 PPE items and body parts (Person, Head, Face, Glasses, Face-mask-medical, Face-guard, Ear, Earmuffs, Hands, Gloves, Foot, Shoes, Safety-vest, Tools, Helmet, Medical-suit, Safety-suit) (**no** negative classes, but Head/Hands/Foot let you infer missing PPE) | SH17, 8,099 images, CC BY-NC-SA 4.0 | Not stated separately | Val (1,620 img): best YOLOv9-e mAP50 70.9, mAP50-95 48.7; YOLOv8-s 63.7 / 41.7 | Latest README entry 2024-11-28 | **Non-commercial** dataset license; needs rule logic for "missing" |
| 8 | [keremberke/yolov8m-protective-equipment-detection](https://huggingface.co/keremberke/yolov8m-protective-equipment-detection) | YOLOv8m | Via `ultralyticsplus` (old Ultralytics 8.0.23) | 10: glove, goggles, helmet, mask, no_glove, no_goggles, no_helmet, no_mask, no_shoes, shoes (yes) | keremberke/protective-equipment-detection (Jan 2023); license not stated | Not stated | Val mAP50 **0.273** | Dataset 2023-01-18 | Very low mAP; stale; not recommended |

## 3. Datasets for fine-tuning

| Dataset (link) | Size | Classes | License | Notes |
|---|---|---|---|---|
| [Ultralytics Construction-PPE](https://docs.ultralytics.com/datasets/detect/construction-ppe) | 1,000 images, 14,863 boxes; 800 train / 200 val, no test split (docs page as fetched) | 12: helmet, gloves, vest, boots, goggles, Person, no_helmet, no_goggle, no_gloves, no_boots, forklift, no_vest | Images and annotations CC BY 4.0; YAML header AGPL-3.0 (per docs) | Built in: `data="construction-ppe.yaml"`, zip at `https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip` (632.3 MB). Has explicit negatives **including boots**. A search snippet of the same page quoted 1,416 images / 11 classes, so check the YAML before relying on the counts |
| [SH17](https://github.com/ahmadmughees/SH17dataset) ([paper](https://arxiv.org/abs/2407.04590)) | 8,099 images, 75,994 instances | 17 (see above) | CC BY-NC-SA 4.0 (README); paper text suggests commercial use is allowed, which conflicts with the README | Images sourced from Pexels; manufacturing focus |
| [Hard Hat Workers (Roboflow)](https://public.roboflow.com/object-detection/hard-hat-workers) | ~7,035–7,041 images (varies by version; [blog](https://blog.roboflow.com/introducing-an-improved-hard-hat-dataset-for-computer-vision-in-workplace-safety/)) | head, helmet, person | Listed as Public Domain by Roboflow (original authors' terms not verified) | Hardhat only; `head` acts as "no helmet" |
| [CHV (ZijianWang-ZW/PPE_detection)](https://github.com/ZijianWang-ZW/PPE_detection) | 1,330 images | person, vest, 4 helmet colours | No license seen on repo; Roboflow mirrors (e.g. [this one](https://universe.roboflow.com/rawabi-aldossary-0vynu/chv-pm6yc)) say CC BY 4.0 (unverified vs original) | Helmet/vest only |
| [Pictor-PPE (ciber-lab/pictor-ppe)](https://github.com/ciber-lab/pictor-ppe) | 774 crowd-sourced + 698 web-mined images | Hat, Vest, Worker ([per SH17 paper](https://arxiv.org/pdf/2407.04590v1)) | **Not found / unverified**; web-mined images may carry third-party copyright | Small and older |
| Roboflow "PPE Combined Model" v4 ([project](https://universe.roboflow.com/roboflow-universe-projects/personal-protective-equipment-combined-model)) | Not verified (page returned 403) | Same taxonomy as the current Hexmon model (per [Hexmon card](https://huggingface.co/Hexmon/vyra-yolo-ppe-detection)) | CC BY 4.0 (per Hexmon card; not verified on Roboflow) | Likely source of the current model's classes |
| Roboflow "Construction Site Safety" family (e.g. [safety-object-detection-app](https://universe.roboflow.com/safety-object-detection-app/construction-site-safety-ccp7o)) | Varies | Includes NO-Hardhat, NO-Safety Vest (per search results) | Several forks list CC BY 4.0; check each page | Many forks with no single canonical version |

**Fine-tune suggestion (estimate, not validated):** start from SafetyVision `v2/best.pt`, or from `yolo11s.pt`/`yolo26s.pt`. Train on a merged, class-remapped mix of the Ultralytics Construction-PPE set (CC BY 4.0, adds boots and no_boots) and the CC BY 4.0 Roboflow sets. Avoid SH17 if the project may ever be used commercially.

## 4. Test video candidates

Licence context:
- **Pexels License** ([pexels.com/license](https://www.pexels.com/license/)): free use and modification, no attribution required. Do not redistribute on other stock or wallpaper platforms. Do not sell unaltered copies. Do not show identifiable people in a bad light.
- **Pixabay Content License** ([summary](https://pixabay.com/service/license-summary/)): free, no attribution required. **Standalone distribution of unmodified content is not allowed**, which makes committing a raw Pixabay clip to a public repo questionable.

Pexels is therefore the better fit for a public GitHub repo, with a credit line in the README. Labelling recognisable workers as "non-compliant" could be read as showing them "in a bad light", so wide or back-view shots are safer.

PPE visibility was **not verified**: the clips were not watched. The descriptions below are the source pages' own text.

| # | Page | Direct file | Author | License | Duration | Resolution | PPE (from page text, unverified) |
|---|---|---|---|---|---|---|---|
| **1 (chosen)** | [Workers Walking in Construction Site (5434223)](https://www.pexels.com/video/workers-walking-in-construction-site-5434223/) | [mp4](https://videos.pexels.com/video-files/5434223/5434223-hd_1920_1080_24fps.mp4) | Everett Bumstead | [Pexels License](https://www.pexels.com/license/) | 0:24 | 1920x1080, ~24 fps | "Workers in safety gear walking through construction site indoors". Gloves and goggles status unknown |
| 2 | [Back View of Workers Walking in Construction Site (5434220)](https://www.pexels.com/video/back-view-of-workers-walking-in-construction-site-5434220/) | [mp4](https://videos.pexels.com/video-files/5434220/5434220-hd_1920_1080_24fps.mp4) | Everett Bumstead | [Pexels License](https://www.pexels.com/license/) | 0:17 | 1920x1080 | "Back view of construction worker walking in safety gear on site". Back view, so faces are not the focus |
| 3 | [Men Working on Construction Site, time-lapse (13921040)](https://www.pexels.com/video/men-working-on-construction-site-13921040/) | [mp4](https://videos.pexels.com/video-files/13921040/13921040-uhd_2560_1440_30fps.mp4) | CESAR A RAMIREZ VALLEJO TRAPHITHO | [Pexels License](https://www.pexels.com/license/) | 0:20 | listed 3840x2160 (file 2560x1440) | "Time-lapse of workers in protective gear building a structure". Wide shot; time-lapse may hurt tracking |
| 4 | [Construction Workers at Construction Site (10294766)](https://www.pexels.com/video/construction-workers-at-construction-site-10294766/) | [mp4](https://videos.pexels.com/video-files/10294766/10294766-hd_1920_1080_30fps.mp4) | This Viktọ | [Pexels License](https://www.pexels.com/license/) | 0:16 | 1920x1080 | "Workers in hardhats reinforcing concrete structures"; vests not mentioned, so it may show hardhat without vest |
| alt | [Work, Worker, Repair (Pixabay 40813)](https://pixabay.com/videos/work-worker-repair-workers-builder-40813/) | [mp4](https://cdn.pixabay.com/video/2020/06/01/40813-426920959_large.mp4) | AlexKopeykin | [Pixabay Content License](https://pixabay.com/service/license-summary/) | ~0:25 (from search listing) | 3840x2160 | No PPE in tags. **Standalone redistribution restriction** |

No clip of exactly ~30 s with clearly *partial* PPE was confirmed. Candidate 1 was chosen as the longest Pexels clip, at 24 s.

**Downloaded:** candidate 1 → `C:\Users\KAREMDAIANEMARCOMINI\portfolio\new\_ppe_candidate_video\pexels-5434223-workers-walking-construction-site.mp4` (18,041,361 bytes, HTTP 200, `video/mp4`). It is not committed to the repo and was not opened. Before committing:
- Watch it to confirm the PPE content.
- Consider trimming or re-encoding it to keep the repo small.
- Add a credit line: "Video by Everett Bumstead on Pexels, https://www.pexels.com/video/5434223/".

## 5. Method and date of research

- Date: **2026-11-09**.
- Tools: web search and page fetches of Hugging Face model cards, GitHub READMEs, Ultralytics docs, Roboflow pages, Pexels and Pixabay pages, and the license pages.
- No model weights were downloaded or run. The only file downloaded was the one video above, fetched with `curl` and a generic user agent.
- The Roboflow Universe project pages returned HTTP 403, so the Roboflow licenses cited come from secondary sources (the HF cards).

## 6. Caveats

- All metrics are **self-reported** and use different datasets and splits (test vs val vs unstated). They are **not comparable** across rows, and none were reproduced here.
- **Licensing:**
  - Most Ultralytics-trained weights inherit **AGPL-3.0** obligations, whatever the card says (MIT, Apache or CC BY labels on candidates 0, 3, 4 and 5 are questionable).
  - SH17 is **non-commercial**.
  - Several dataset licenses could not be verified at the original source.
  - Get licensing reviewed before any commercial or networked deployment.
- **Contradictions:**
  - The Ultralytics Construction-PPE page gave different image and class counts in a search snippet (1,416 / 11) and in the fetched page (1,000 / 12).
  - SH17's README license conflicts with its paper.
- The video's PPE content and framing were not visually verified. Pexels "Free" was read from the page; the full terms are at [pexels.com/license](https://www.pexels.com/license/).
- **Privacy:** worker footage is personal data. Prefer wide or back shots, avoid identifying individuals, and keep humans in the loop. The model cards above also state that the models should not drive disciplinary decisions on their own.
- PPE detector output is a screening signal, not evidence. A qualified person should review results before any safety decision.
