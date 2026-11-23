# Demo GIFs

Screen recordings used in the main README, all made on the clips in
[`samples/`](../../samples/README.md) (sources and licenses are listed there).
They are illustrations of the method, not evaluations.

| File | Content |
|---|---|
| `demo.gif` | Detection only: annotated output of the pipeline on the construction-site segment (see [`results/`](../../results/README.md)). |
| `demo_app.gif` | Screen recording of the Streamlit app processing a 12 s excerpt of the construction-site clip, with the infraction filter and report cards (processing and VLM step sped up). |
| `demo_vlm.gif` | Excerpt (check 2 of 3, person #5: analysis counter, then the SmolVLM-500M answer) of [`results/demo_vlm_doublecheck.mp4`](../../results/demo_vlm_doublecheck.mp4), rendered by `scripts/make_demo_video.py` from `samples/construction_site.mp4` and `samples/compliant_worker.mp4`. |

Detections and VLM answers shown are model outputs and require human review.
