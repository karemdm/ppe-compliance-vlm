# Sample input

`ppe_demo_input.mp4` (1920x1080, 30 fps, ~53 s, no audio) is a test input for the app.
It concatenates four real-time stock clips, re-encoded to H.264:

| Segment | Source | License | Content (checked on preview frames) |
|---|---|---|---|
| 0–20 s | [Pexels 14117667](https://www.pexels.com/video/men-at-the-construction-site-14117667/) "Men at the Construction Site" by CESAR A RAMIREZ VALLEJO TRAPHITHO (first 20 s) | [Pexels License](https://www.pexels.com/license/) | Wide shot of about 20 workers pouring a concrete slab. Most wear hardhats, almost nobody wears a hi-vis vest, one wears a sun hat instead of a hardhat, gloves are mixed. |
| 20–30 s | [Pexels 4281405](https://www.pexels.com/video/4281405/) "Men Working in a Warehouse" | [Pexels License](https://www.pexels.com/license/) | High-angle shot of three workers in a warehouse aisle: no hardhats, no vests, two of them wear gloves. |
| 30–38 s | [Pexels 8964379](https://www.pexels.com/video/engineers-in-a-construction-site-having-a-conversation-8964379/) "Engineers in a Construction Site Having a Conversation" by Mikael Blomkvist | [Pexels License](https://www.pexels.com/license/) | Two workers at an industrial steel structure. Woman: hardhat and hi-vis vest, no gloves. Man: hardhat and gloves, no vest. |
| 38–53 s | [Pexels 4727768](https://www.pexels.com/video/video-footage-of-hardware-warehouse-4727768/) "Video Footage of Hardware Warehouse" by Luday Leguen (first 15 s) | [Pexels License](https://www.pexels.com/license/) | Lumber warehouse with reach-truck forklifts: a spotter with a hi-vis vest and no hardhat, an operator with neither. |

The author of the 4281405 clip could not be confirmed automatically (the Pexels page blocks scripted access); see its link for the credit.

PPE per worker is a visual estimate made while selecting the clips (with AI assistance), not ground truth.
The clips are used only to illustrate the method. Detections produced on them are model outputs, not a judgement about the people shown.
