# Results

Outputs of `python -m ppe_compliance analyze --no-vlm` with the default settings (`safetyvision`
model, person `conf=0.45`, PPE `conf=0.15`, 3 s temporal confirmation), on CPU.
These runs were made **before person re-identification was added**: persons are tracker ids
and the JSON files have no `track_ids` / `reid_summary`. Re-running the commands below with
the current code gives re-identified results.
Annotated videos are re-encoded to 720p and **played at 1.5× speed**; processing ran at the original speed.
Person boxes: gray while checking, green when compliant, red with the confirmed violations.

## Per-clip runs

| Sample | Annotated video | Report | Confirmed violations (tracked IDs) |
|---|---|---|---|
| [`construction_site.mp4`](../samples/construction_site.mp4) | [`construction_site_annotated.mp4`](construction_site_annotated.mp4) | [`construction_site_report.json`](construction_site_report.json) | 15 IDs: 15 × NO-Safety Vest, 3 × NO-Hardhat |
| [`industrial_structure.mp4`](../samples/industrial_structure.mp4) | [`industrial_structure_annotated.mp4`](industrial_structure_annotated.mp4) | [`industrial_structure_report.json`](industrial_structure_report.json) | 2 IDs: 1 × NO-Safety Vest, 1 × NO-Goggles |
| [`carpenters.mp4`](../samples/carpenters.mp4) | [`carpenters_annotated.mp4`](carpenters_annotated.mp4) | [`carpenters_report.json`](carpenters_report.json) | 3 IDs: 3 × NO-Safety Vest |
| [`warehouse_aisle.mp4`](../samples/warehouse_aisle.mp4) | [`warehouse_aisle_annotated.mp4`](warehouse_aisle_annotated.mp4) | [`warehouse_aisle_report.json`](warehouse_aisle_report.json) | 3 IDs: 3 × NO-Safety Vest, 2 × NO-Hardhat, 2 × NO-Goggles |
| [`compliant_worker.mp4`](../samples/compliant_worker.mp4) | [`compliant_worker_annotated.mp4`](compliant_worker_annotated.mp4) | [`compliant_worker_report.json`](compliant_worker_report.json) | none (worker wears hardhat and vest) |

## VLM double-check demo

[`demo_vlm_doublecheck.mp4`](demo_vlm_doublecheck.mp4) (62 s, 720p) and [`demo_vlm_doublecheck.json`](demo_vlm_doublecheck.json):
construction site + compliant worker, with three SmolVLM-500M double-checks (the video freezes while the VLM runs;
waits are shortened, real times are shown and logged). Built with `scripts/make_demo_video.py`.

## Combined run

[`ppe_demo_safetyvision.mp4`](ppe_demo_safetyvision.mp4) and [`report_safetyvision.json`](report_safetyvision.json):
the six-clip [`ppe_demo_input.mp4`](../samples/ppe_demo_input.mp4) (~83 s), described in the main README.

Reproduce one clip:

```bash
python -m ppe_compliance analyze --input samples/construction_site.mp4 \
    --output results/construction_site_report.json --save-video outputs/construction_site.mp4 --no-vlm
```

In these runs counts are per tracked ID, so tracker ID switches can count the same worker more
than once; with re-identification (current default) they are merged per person when the
appearance matches.
These are model outputs on a handful of clips, not a validated evaluation; review them before drawing conclusions.
