# CLIP Disaster Damage Assessment

Using OpenAI's CLIP model to detect and localize visual change in pre/post-disaster satellite imagery, applied to Hurricane Helene and Hurricane Milton's impact on Treasure Island, FL (2024).

## Overview

This project compares three independent pre/post image pairs from different satellite sources — Maxar, NOAA, and Sentinel-2 — covering the same coastal region before and after two consecutive hurricanes. Rather than relying on a single whole-image similarity score, the pipeline also performs **patch-level comparison**, splitting each image into a 4×4 spatial grid to localize *where* visual change is concentrated rather than just *how much* change occurred overall.

## Method

1. Load each image pair and extract CLIP image embeddings (`openai/clip-vit-base-patch32`) via HuggingFace Transformers
2. Compute whole-image cosine similarity between the pre and post embeddings
3. Resize both images in a pair to a common resolution (1024×1024) to ensure spatial alignment
4. Split each into a 4×4 grid (16 patches) and compute cosine similarity independently per corresponding patch
5. Visualize results as side-by-side image comparisons with an overlaid "change heatmap" (1 − similarity)

## Results

| Pair | Whole-Image Cosine Similarity |
|------|-------------------------------|
| Maxar (Pre-Nov 2023 vs Post-Oct 2024) | 0.928 |
| NOAA (Post-Helene vs Post-Milton) | 0.959 |
| Sentinel-2 (Pre-Sep 2024 vs Post-Oct 2024) | 0.869 |

<img width="1489" height="1481" alt="image" src="https://github.com/user-attachments/assets/ca499a56-094e-4bb3-8f0a-e36eef534924" />


## Key Findings & Limitations

- **Sentinel-2** shows the lowest whole-image similarity (most apparent change), but patch-level analysis reveals this is largely driven by **cloud cover present in the pre-event image and absent in the post-event image**, not necessarily ground damage.
- **NOAA** images contain differing extents of no-data (black) capture regions between the two dates, producing localized false-positive "change" signals unrelated to storm impact.
- **Maxar** is cloud-free and higher resolution, likely the most reliable signal of the three, though it also shows some elevated change near its own no-data border region — a capture artifact rather than true change.
- **Takeaway:** General-purpose CLIP embeddings are sensitive to *any* visual difference — including clouds, sensor artifacts, and capture misalignment — not just genuine ground change. A production-ready pipeline would need cloud-masking and no-data handling before similarity scoring to isolate true damage signals.
