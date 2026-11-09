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

See Round 2 below.

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

**Outcome:** candidate 1 was downloaded and reviewed frame by frame: both workers wear a hardhat and a vest, so it does not show missing PPE and was **not used**. The demo input finally used is described in [`samples/README.md`](../samples/README.md): Pexels 13921040 (construction site, mixed hardhat/vest compliance) followed by Pexels 4281405 (warehouse, no hardhats or vests), about 30 s in total.

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

## Round 2 (deeper search, 2026-11-09)

> Prepared with AI assistance (Claude) on 2026-11-09. All figures are copied from the cited pages. They are self-reported by the authors and were **not** reproduced here. No weights were downloaded. HF pages often show only relative dates ("5 days ago"), which are relative to 2026-11-09.

### R2.1 Coverage (queries and sources)

- **Hugging Face model search:**
  - [ppe by downloads](https://huggingface.co/models?search=ppe&sort=downloads), [ppe by likes](https://huggingface.co/models?search=ppe&sort=likes) and [ppe by recently modified](https://huggingface.co/models?search=ppe&sort=modified)
  - [hardhat](https://huggingface.co/models?search=hardhat&sort=downloads), [hard-hat](https://huggingface.co/models?search=hard-hat&sort=downloads), [helmet](https://huggingface.co/models?search=helmet&sort=downloads) and [no-helmet](https://huggingface.co/models?search=no-helmet&sort=downloads)
  - [safety vest](https://huggingface.co/models?search=safety+vest&sort=downloads), [construction safety](https://huggingface.co/models?search=construction+safety&sort=downloads), [protective equipment](https://huggingface.co/models?search=protective+equipment&sort=downloads), [safety gear](https://huggingface.co/models?search=safety+gear&sort=downloads) and [ppe detr](https://huggingface.co/models?search=ppe+detr) (covers DETR, RT-DETR and RF-DETR)
  - Only page 1 of each listing was read. The hardhat and helmet listings are dominated by DETR tutorial fine-tunes and generative or LLM repos, which were skipped.
- **GitHub:**
  - [ppe detection yolo, by stars](https://github.com/search?q=ppe+detection+yolo&type=repositories&s=stars&o=desc), [ppe yolo26, by stars](https://github.com/search?q=ppe+yolo26&type=repositories&s=stars&o=desc), [hard hat detection, by stars](https://github.com/search?q=hard+hat+detection&type=repositories&s=stars&o=desc) and [construction site safety yolo, by recently updated](https://github.com/search?q=construction+site+safety+yolo&type=repositories&s=updated&o=desc)
  - The [ppe-detection topic](https://github.com/topics/ppe-detection?o=desc&s=stars)
  - A web search for YOLO11/YOLO26 repos with `no-helmet`/`no-vest` classes and released `best.pt`
- **Roboflow Universe:**
  - A web search for PPE projects with `no-helmet`/`no-vest` classes.
  - Project pages returned HTTP 403 again (for example [ppe-detection-using-cv](https://universe.roboflow.com/ai-camp-safety-equipment-detection/ppe-detection-using-cv)).
  - The weight-download policy was checked in the [Roboflow docs](https://docs.roboflow.com/deploy/download-roboflow-model-weights), along with the [Roboflow PPE blog](https://roboflow.com/blog/ppe-detection).
- **Papers:**
  - Papers with Code now redirects to HF Papers ([paperswithcode.com/task/ppe-detection](https://paperswithcode.com/task/ppe-detection) returns a 302 to huggingface.co/papers/trending).
  - Web searches covered arXiv/MDPI 2024–2026 PPE papers and SH17 follow-ups.
- **Ultralytics:** the [Construction-PPE docs](https://docs.ultralytics.com/datasets/detect/construction-ppe/), the [dataset YAML](https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/construction-ppe.yaml) and the [missing-PPE blog](https://www.ultralytics.com/blog/missing-ppe-detection-yolo).

### R2.2 New candidates with paired classes and downloadable weights

The rows are ordered by how much usable published evidence each one has, not by expected quality.

| # | Model (link) | Arch | Weights | Classes (pairs) | Training data / license | Weights license | Published metrics (split) | Last update | Red flags |
|---|---|---|---|---|---|---|---|---|---|
| N1 | [constructelligence/construction-site-safety-hazards](https://huggingface.co/constructelligence/construction-site-safety-hazards) | YOLOv8s @640 | `buildvision-hazards-v0.3.pt` (22.6 MB) and `.onnx` (44.8 MB), [files](https://huggingface.co/constructelligence/construction-site-safety-hazards/tree/main) | 14 listed: person, hardhat/no-hardhat, safety vest/no-safety vest, gloves/no-gloves, goggles/no-goggles, no-mask (no positive mask class), safety shoes (no "no-shoes" class), excavator, wheel loader, dump truck | 7,450 train / 567 val images from 5 Roboflow Universe sources (Construction Site Safety, construction-safety-gsnvb, PPE detection 1, PPE_Dectection v4, excavators-czvg9), stated as CC BY 4.0 | AGPL-3.0 | **Own test set** (573 img): mAP50 0.634, mAP50-95 0.373, P 0.754, R 0.604. Per-class mAP50: no-hardhat 0.379, no-safety vest 0.522, no-goggles 0.208, no-gloves 0.074, no-mask 0.000. **Held-out benchmark** (284 img = Construction-PPE val+test, "core" = person/hardhat/no-hardhat/vest): 0.642, vs killuminati1 0.543, wesleymqsss 0.515, Hansung-Cho 0.503, melihuzunoglu 0.435, jashwanthpeddisetty 0.345, Hexmon 0.286, baskarmother 0.180 | v0.3 about 1 month ago, card edited 24 days ago ([files](https://huggingface.co/constructelligence/construction-site-safety-hazards/tree/main)); benchmark dated October 2026 | Benchmark run by the author. Test splits come from the same sources as training. Some person boxes were pseudo-labelled with COCO YOLOv8s. Weak no-gloves, no-goggles and no-mask. The commit message says "15 classes" but the card lists 14. **SafetyVision and CHD were not in its benchmark** |
| N2 | [badmuriss/ppe-detection-yolov8s](https://huggingface.co/badmuriss/ppe-detection-yolov8s) | YOLOv8s-P2 (extra stride-4 head) @960 | `best.pt` 43.2 MB ([files](https://huggingface.co/badmuriss/ppe-detection-yolov8s/tree/main)) | 4: helmet, vest, head (meaning no helmet), no_vest | 6 Roboflow Universe sets (PPE Combined Model, Hard Hats, Hard Hat Universe, Safety Vests, vest-qf3av, vest-pbrbu), 36,574 images after dedup. Per-source licenses not listed | AGPL-3.0 | **Val:** mAP50 0.806, mAP50-95 0.556. **Test** (1,828 img) per-class mAP50: helmet 0.975, vest 0.901, head 0.939, no_vest 0.650 (only 20 GT boxes) | 3 months ago ([files](https://huggingface.co/badmuriss/ppe-detection-yolov8s/tree/main)) | Hardhat and vest only. no_vest is statistically thin. Recall falls sharply on small objects (no_vest recall 0.090 at 0.2x scale). Real-domain recall not measured (per card) |
| N3 | [jashwanthpeddisetty0712/ppe-detection-yolov8m](https://huggingface.co/jashwanthpeddisetty0712/ppe-detection-yolov8m) | YOLOv8m @640 | `best.pt` referenced in the card (file list not verified) | Same 14 classes as Hexmon (5 pairs + Fall-Detected, Ladder, Safety Cone, Person) | Not named; no license | AGPL-3.0 | **Val** (22,077 instances): mAP50 0.79, mAP50-95 0.53. NO-Safety Vest mAP50 0.21, NO-Gloves 0.91, NO-Goggles 0.96. On N1's benchmark: core 0.345 | Jul 24 ([listing](https://huggingface.co/models?search=ppe&sort=downloads)) | Probably trained on the same PPE-Combined data as Hexmon (inferred from the identical taxonomy, unverified). Validation only |
| N4 | [abdullahzunorain/rt-detr-v2-fine-tuned-PPE-Det](https://huggingface.co/abdullahzunorain/rt-detr-v2-fine-tuned-PPE-Det) | RT-DETRv2 R50vd (HF Transformers), input 800x1333 | `best_model.pt` (PyTorch checkpoint, not Ultralytics) | 11: person; helmet, coverall, boot, glove and eyewear each `_present`/`_absent` | 3,200 CCTV images from **one facility**; no license stated | MIT tag (base-model license not stated) | **Val** (964 img): mAP50 0.623, mAP50-95 0.397 | May 16 ([listing](https://huggingface.co/models?search=ppe+detr)) | Single-site data. Not Ultralytics-compatible (needs a Transformers loader). Weak glove and eyewear classes |
| N5 | [haroonx02/construction-ppe-yolo11n](https://huggingface.co/haroonx02/construction-ppe-yolo11n) | YOLO11n @640 | `yolo11n_ppe.pt` 5.5 MB, plus ONNX FP32/FP16/INT8 | 10: Hardhat/NO-Hardhat, Mask/NO-Mask, Safety Vest/NO-Safety Vest, Person, Safety Cone, machinery, vehicle | Roboflow "Construction Site Safety" (2,605 train images), CC BY 4.0 | AGPL-3.0 | **Test** (82 img): mAP50 0.713, mAP50-95 0.413. "Clean" 40-image subset: 0.671 / 0.384 | 6 days ago ([listing](https://huggingface.co/models?search=ppe&sort=modified)) | Author says half the test images share source videos with training. Tiny test set. About 1/3 of violators missed. Labelled "a learning project, not a safety system" |
| N6 | [TechsphereAI/FactoryVision-PPE-YOLO](https://huggingface.co/TechsphereAI/FactoryVision-PPE-YOLO) | YOLO11n @640 | `model.pt` | 10 classes, names not on the card (listed in `labels.json`, not checked) | Construction Site Safety (Snehil Sanyal / Roboflow, CC BY 4.0 per card) | **None assigned** (card warns about AGPL) | **Test** (82 img): mAP50 0.705, mAP50-95 0.393, R 0.640 | 5 days ago ([listing](https://huggingface.co/models?search=ppe&sort=modified)) | Educational prototype. Same small test set as N5 |
| N7 | [mancalazure/construction-ppe-safety](https://huggingface.co/mancalazure/construction-ppe-safety) | YOLOv8n @640 | `best.pt`, plus ONNX for DeepStream/Triton | Same 10 classes as N5 | Roboflow construction-site-safety v28 + snehilsanyal set | CC BY 4.0 (card; questionable for Ultralytics-trained weights) | **None** ("No validation metrics are published here") | Sep 10 ([listing](https://huggingface.co/models?search=ppe&sort=downloads)) | No metrics. NO-Hardhat misses caps, beanies and hoods (per card) |
| N8 | [wesleymqsss/yolov8m-construction-site-safety](https://huggingface.co/wesleymqsss/yolov8m-construction-site-safety) | YOLOv8m | `best.pt` referenced (unverified) | Same 10-class Construction Site Safety taxonomy (Portuguese names in card) | Kaggle snehilsanyal set (2,834 img); license not stated | MIT tag | Per-class table, **split not named** (131-image test set). On N1's benchmark: core 0.515 | Sep 20 ([listing](https://huggingface.co/models?search=construction+safety&sort=downloads)) | Unfinished card; tiny test set |
| N9 | [thalostech2025/dfine-small-construction-ppe-v1](https://huggingface.co/thalostech2025/dfine-small-construction-ppe-v1) | D-FINE small (Transformers, 10.3M params) | safetensors | 12: Person, Hardhat/No-hardhat, Safety-vest/No-safety-vest, No-mask, No-goggles, Gloves, Safety-boots, 3 vehicles | `thalostech2025/construction-ppe-equipment-v1`, license not stated | Apache-2.0 | "Evaluation set": mAP50 0.460. No-safety-vest mAP 0.001, No-mask 0.000, No-goggles 0.000 | Jun 29 ([listing](https://huggingface.co/models?search=ppe&sort=downloads)) | Negative classes essentially non-functional |

**GitHub repos with paired classes and weights in the repo.** All are small, and their numbers come from Construction-PPE runs.

| Repo | Arch | Pairs | Weights | Metrics | License | Red flags |
|---|---|---|---|---|---|---|
| [AbdullahMG/ppe-detection](https://github.com/AbdullahMG/ppe-detection) | YOLO11n | helmet, goggles, gloves, boots (**no no_vest**) | `models/` folder; whether weights are committed is unconfirmed | Test (141 img, old Construction-PPE): mAP50 0.550, mAP50-95 0.265 | Repo license not stated | 0 stars, 4 commits |
| [C-Nekopedia/SiteGuard](https://github.com/C-Nekopedia/SiteGuard) | YOLO26n | same 4 pairs (no no_vest) | `data/models/yolo26n_ppe.pt` per README (not verified) | mAP50 0.523, mAP50-95 0.270, split not stated | MIT code, weights not stated | 7 stars |
| [evgomes/yolo26-construction-ppe-image-recognition](https://github.com/evgomes/yolo26-construction-ppe-image-recognition) | YOLO26n | same 4 pairs | `runs/train/ppe_yolo26n/weights/best.pt` | None published | Not shown | 1 star |
| [BulletsHo/ppe-sentinel-yolo26](https://github.com/BulletsHo/ppe-sentinel-yolo26) | YOLO26 (n inferred) | 28 classes, only 6 named, so pairing unconfirmed | `outputs/.../best.pt` in repo | None in README | AGPL-3.0+ | Class list not visible |
| [Ansarimajid/Construction-PPE-Detection](https://github.com/Ansarimajid/Construction-PPE-Detection) | YOLOv8 | Hardhat, Mask and Safety Vest pairs | `Model/ppe.pt` | None | MIT (code) | 51 stars. A `.env` file appears in the public tree; contents not examined. If the owner is contacted, it should be flagged to them |
| [snehilsanyal/Construction-Site-Safety-PPE-Detection](https://github.com/snehilsanyal/Construction-Site-Safety-PPE-Detection) | YOLOv8n | Hardhat, Mask and Safety Vest pairs | `models/best.pt` | Plots only, no numbers | Not stated | 212 stars ([topic page](https://github.com/topics/ppe-detection?o=desc&s=stars)), 1 commit, last updated Aug 18, 2024 |

**Also useful as evidence:** [nusRying/ai-safety-monitoring](https://github.com/nusRying/ai-safety-monitoring) (proprietary, no weights released) reports a YOLO26n trained for 50 epochs on the older 1,416-image Construction-PPE. On the test split (Ultralytics scoring) it got mAP50 0.549 overall, but **no_helmet 0.236, no_goggle 0.221, no_gloves 0.113 and no_boots 0.035**, against helmet 0.935 and vest 0.910.

**Seen and excluded:**
- Positive-only or two-class models:
  - [Subh775/yolo26s-hardhat](https://huggingface.co/Subh775/yolo26s-hardhat): YOLO26s; head/helmet on Hard Hat Workers v5; test mAP50 0.980, mAP50-95 0.688; AGPL-3.0. It is a good hardhat-only specialist, where `head` acts as "no helmet".
  - [leeyunjai/yolo26-helmet](https://huggingface.co/leeyunjai/yolo26-helmet): helmet / "face (no helmet)"; no license or dataset stated.
- No metrics or wrong card:
  - [leeyunjai/yolo11-ppe](https://huggingface.co/leeyunjai/yolo11-ppe): YOLO11x; helmet/no helmet/vest/no vest/person; card copied from a "number and operator" model; no metrics or license.
  - [baskarmother/yolov8-ppe-construction](https://huggingface.co/baskarmother/yolov8-ppe-construction): no metrics.
  - [akashpannala/ppe-compliance-yolov8](https://huggingface.co/akashpannala/ppe-compliance-yolov8): empty card.
  - [sanrapi/rf-detr-finetuned-ppe](https://huggingface.co/sanrapi/rf-detr-finetuned-ppe): no card.
- Not suitable:
  - [qualcomm/PPE-Detection](https://huggingface.co/qualcomm/PPE-Detection): 2 output classes, proprietary dataset, no accuracy metrics, no `.pt` (ONNX/TFLite/QNN only).
  - [arielshakaramiro/ppe-vision](https://huggingface.co/arielshakaramiro/ppe-vision): weights not published, "all-rights-reserved", trained partly on SH17.
- **Security note:** [thalostech2025/thalos-industrial-ppe-v1](https://huggingface.co/thalostech2025/thalos-industrial-ppe-v1) has no metrics. Its card loads pickled weights with `torch.load` from a URL, and it ends with a section ("PHASE 4 — Ready for Cursor") written as instructions to an AI assistant. That text was treated as data and ignored. It is a possible prompt-injection pattern and should be reported to IT; do not use this repo.
- Many HF repos updated in the last two weeks (`Targaryen101/ppe_*`, `CVCompressor/ppe_*`, `ScoreVision_ppe01`, `manako/...probe-set-winner` and others, [listing](https://huggingface.co/models?search=ppe&sort=modified)) have no cards. They look like automated competition uploads and were not evaluated.

### R2.3 Roboflow, papers and Ultralytics

- **Roboflow Universe:**
  - No PPE project was found with a public direct `.pt` download.
  - Roboflow's docs describe weight download only for models in **your own workspace**: Core/Free tier through the Python SDK, in `.pt` and ONNX among other formats. The page does not say that other users' public Universe models can be downloaded ([docs](https://docs.roboflow.com/deploy/download-roboflow-model-weights)).
  - The same page notes that running AGPL YOLO weights commercially outside Roboflow needs an Enterprise license or AGPL compliance.
  - The practical route is to fork a Universe dataset and train it yourself.
  - The Roboflow PPE blog (May 18, 2026) trains RF-DETR Small and reports mAP50 79.2%, but it does not give a downloadable weight link, and its model link title ("traffic-light-detection") does not match ([blog](https://roboflow.com/blog/ppe-detection)).
  - RF-DETR is mostly Apache-2.0, but some sizes fall under a Platform Model License (per a web search of Roboflow's [license page](https://roboflow.com/model-licenses/rf-detr); the sources conflict and this is unverified).
- **Papers 2024–2026:**
  - No 2025–2026 PPE paper with **released weights** and paired classes was found.
  - SH17's journal version is in the *Journal of Safety Science and Resilience*, June 2025 issue ([DOAJ](https://doaj.org/article/d5155a83f94942b299109f67d9f9abea)). No follow-up release was found.
  - A 2026 MDPI *Buildings* study on real/synthetic data mixing compares YOLO11s and RT-DETR-L ([doi](https://doi.org/10.3390/buildings16102034); full text returned 403, so whether code or weights were released is unverified).
  - An ETASR low-light PPE violation dataset (11,407 images, with compliance/non-compliance pairs) was seen only in a search snippet ([link](https://etasr.com/index.php/ETASR/article/view/14254)); weights and license are unverified.
- **Ultralytics Construction-PPE:**
  - The current docs give 1,000 images (800 train / 200 val, no test split), 12 classes including `no_vest` (id 11), and CC BY 4.0 for images and annotations ([docs](https://docs.ultralytics.com/datasets/detect/construction-ppe/), [YAML](https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/construction-ppe.yaml)).
  - The blog says it was released on Nov 3, 2026, in Ultralytics 8.4.200 ([blog](https://www.ultralytics.com/blog/missing-ppe-detection-yolo)).
  - **Ultralytics does not publish a pretrained model on it.** Its examples train from `yolo26n.pt` ([docs](https://docs.ultralytics.com/datasets/detect/construction-ppe/), [blog](https://www.ultralytics.com/blog/missing-ppe-detection-yolo)).
  - **Version conflict.** Several 2025–2026 repos describe an earlier version with 1,416 images (1,132 / 143 / 141), 11 classes including `none` and **no `no_vest`** ([AbdullahMG](https://github.com/AbdullahMG/ppe-detection), [SiteGuard](https://github.com/C-Nekopedia/SiteGuard), [nusRying](https://github.com/nusRying/ai-safety-monitoring)). This explains the round-1 contradiction.
  - The docs text and staff comments also say there is no missing-vest class, which contradicts the YAML ([docs](https://docs.ultralytics.com/datasets/detect/construction-ppe/)).
  - Confirm which version the downloaded zip contains before using it.

### R2.4 Verdict: is anything likely better than SafetyVision v2 / CHD?

**No candidate shows published evidence that it is better.**

- **N1 (constructelligence)** is the most interesting new find.
  - It covers almost the same pairs as SafetyVision: hardhat, vest, gloves and goggles, plus safety shoes.
  - It publishes test-split metrics and a cross-dataset benchmark on Construction-PPE, which is more than most cards offer. In that benchmark it beats Hexmon (0.642 vs 0.286 core mAP50), the current model in round 1, and 7 other models ([source](https://huggingface.co/constructelligence/construction-site-safety-hazards)).
  - But its own-test mAP50 (0.634) is lower than SafetyVision v2's (0.766) on a different test set.
  - Its negative classes are weak: no-gloves 0.074, no-goggles 0.208, no-hardhat 0.379, no-mask 0.
  - SafetyVision and CHD were not in its benchmark.
  - Treat it as a candidate worth adding to a head-to-head test, not as a replacement.
- **N2 (badmuriss)** has strong hardhat/head numbers on its own test split, but only 4 classes, and its no_vest result rests on 20 boxes. It is a possible hardhat/vest specialist alternative to CHD. It is not better on evidence, because the datasets differ.
- **The rest** (N3–N9 and the GitHub repos) have validation-only, tiny-test or no metrics, or they show that the negative classes fail.
- **Uncertainty:** every number above comes from a different dataset and split (own test, own val, Construction-PPE val+test, 82-image or 141-image tests), so **none are comparable**.
  - SafetyVision's 0.766 and CHD's val 0.60–0.67 mAP50-95 are also on their own data.
  - Only a shared benchmark can rank these models: the same images, a class remap, and the same imgsz.
  - The current choices stay in place on the strength of their documentation, not because they were proven better.

### R2.5 Is fine-tuning on Construction-PPE (+ CC BY sets) the more reliable path?

**Not on Construction-PPE alone.**

- It is small (1,000 images in the current version).
- Published runs on the earlier version give weak negative classes: overall test mAP50 of 0.523–0.550 at nano size, and no_boots 0.035 / no_gloves 0.113 ([nusRying](https://github.com/nusRying/ai-safety-monitoring), [AbdullahMG](https://github.com/AbdullahMG/ppe-detection), [SiteGuard](https://github.com/C-Nekopedia/SiteGuard)).
- Its real value is as a **CC BY 4.0, independent yardstick**: none of SafetyVision, CHD or N1 says it trained on it (for N1, per its [card](https://huggingface.co/constructelligence/construction-site-safety-hazards)).

**Suggested plan (estimate, not validated):**
1. **Benchmark first.** Remap Construction-PPE val (and test, if present) to the app taxonomy. Run `model.val()` at each model's native imgsz on SafetyVision v2, CHD YOLO26s/l and N1, and optionally N2 for hardhat/vest. Report per-pair recall and precision on the `no_*` classes, then add a hand-labelled clip set from the target domain. This alone may settle the default.
2. **Fine-tune only if the benchmark shows a gap.**
   - Start from the best checkpoint (SafetyVision `v2/best.pt`, or `yolo26s.pt`).
   - Train on a merged, class-remapped mix of:
     - Construction-PPE train
     - CC BY 4.0 Roboflow sets, re-checking each license on its page (N1's list is a starting point: Construction Site Safety, construction-safety-gsnvb, PPE detection 1, PPE_Dectection v4)
     - Hard Hat Workers (`head` as no-helmet)
   - Deduplicate across sources (perceptual hash) so the evaluation images don't leak into training.
3. **Keep a fixed held-out set** that is never trained on: Construction-PPE val plus the target clips. Use it as acceptance criteria, for example minimum recall per `no_*` class, set by the project owner.
4. **Weight the negative classes.** Oversample images that contain `no_*` boxes. Consider rule-based "missing PPE" inference from person + absent item as a fallback for weak pairs (gloves, goggles, boots).
5. **Record the outcome.** Write down dataset licenses (attribution for CC BY), the AGPL-3.0 status of the resulting weights, and the fact that the results are a screening signal that needs qualified human review.
