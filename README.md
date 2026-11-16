# Identifying solar filaments in images of the Sun

A submission to the Kaggle Solar Filament Segmentation Challenge 2026 by
**[Shep Scientific](https://www.kaggle.com/shepscientific)** (team `shepsci`).

Solar filaments appear as dark structures against the Sun in H-alpha images.
This solution identifies individual filaments and assigns each one its own pixel mask.
The selected submission scored **0.40 public Panoptic Quality (PQ)**:
1,317 masks across 178 of 180 processed images. Two images produced no masks.

**Start with [the illustrated notebook](pipeline.ipynb) or
[the four-page technical report](report.pdf).**

## Contents

| Material | Purpose |
| --- | --- |
| [pipeline.ipynb](pipeline.ipynb) | Educational walkthrough, inference, output validation, and optional diagnostics and training |
| [report.pdf](report.pdf) / [report-source/](report-source/) | Final submitted paper and its Overleaf-compatible source |
| [inference.py](inference.py) / [validate.py](validate.py) | Image-to-mask prediction and submission checks |
| [weights/](weights/) | Both required models and their provenance/checksum manifest |
| [requirements.txt](requirements.txt) / [SECURITY.md](SECURITY.md) | Recorded reproduction environment and security guidance |
| [provenance.json](provenance.json) / [validation.json](validation.json) | Model history disclosures and exact submitted-output receipt |
| [architecture.json](architecture.json) | Stored network structures |
| [evaluate.py](evaluate.py) / [diagnostics.json](diagnostics.json) | Descriptive training-set diagnostics, baselines, and uncertainty |
| [train.py](train.py) / [new-training-split.json](new-training-split.json) | Optional new training recipe and observation-grouped split |
| [NOTICE](NOTICE) / [LICENSE](LICENSE) | Attribution and licensing |

## Reproduce the submitted method

Use a dedicated Python environment. The recorded run used **Python 3.12.12 and
Apple MPS**. The exact package snapshot is retained for reproducibility;
[known dependency advisories](SECURITY.md) must be considered before installing it.

1. Install Git LFS, clone this repository, and run `git lfs pull`.
2. Create and activate a Python environment, then run `pip install -r requirements.txt`.
3. Obtain the official competition images under their original licence.
4. Open `pipeline.ipynb`, set `IMAGES` to `test/test_images`, and run the cells.
   For command-line inference:

   ```bash
   python inference.py --images /path/to/test/test_images --device mps
   ```

5. Read `work/submission.csv` and its validation receipt. Compare its checksum with
   [validation.json](validation.json).

Both model files are included through Git LFS (286,752,617 bytes in total).
No external model mirror is required. CUDA and CPU are supported by the code,
but exact byte equality was verified only in the recorded MPS environment.
Backend differences can change predictions near numerical thresholds.

## Method

YOLO26x-seg and YOLOv8x-seg each process the image at 2048 pixels, with confidence
0.5, box-NMS IoU 0, and at most 100 detections. We pool their proposed instance
masks, suppress duplicates above mask IoU 0.5, and assign shared pixels in
descending confidence order. Regions below five pixels are discarded after
pixel assignment. Available predictions are retained when one model detects nothing.
Inference uses only the supplied image.

## Provenance and learning extensions

NOTICE credits HDJoJo for the released weights and initial inference design.
The upstream model cards declare training on the 707 competition training images.
We verified the released versions, checksums, readable source, and recovered
training arguments; the complete original training history was not independently
reconstructed. That limitation is preserved in the report and provenance record.

The optional diagnostics use training observations already seen by the models;
they are descriptive measurements, not held-out validation. The optional training
recipe starts a new network with a fixed split of 566 training and 141 validation
observations. It is a learning extension, not a reconstruction of the submitted
weights. Both optional steps are disabled by default. Training requires its own
storage and compute budget.

## Submitted version

The exact competition package is preserved at
[`competition-submission-2026`](https://github.com/shepsci/filament-segmentation-2026-final/tree/competition-submission-2026),
commit `0db8f251fe6ca45677a7d988d24900b6506ffcc9`, for Kaggle submission **58351849**.
The paper was named `report-review.pdf` in that snapshot; `report.pdf` contains
the identical bytes. Later documentation cleanup changes no prediction code,
model weights, recorded environment, report content, or measured result.
