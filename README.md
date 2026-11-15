# Solar Filament Segmentation 2026 - shepsci

PRIVATE REVIEW COPY. Publication requires completed owner review and explicit approval. Do not automatically publish at the deadline.

This package reproduces our final two-model filament segmentation method.
Start with `pipeline.ipynb`. It loads the model weights stored in this repository
using Git LFS and checks that the files match the recorded originals.
Supply the official competition images under their original licence.

## Find the final materials

- `report-review.pdf`: the current four-page paper.
- `report-source/`: its Overleaf-compatible LaTeX source and figures.
- `pipeline.ipynb`: the complete illustrated workflow, with optional training and evaluation.
- `inference.py` and `validate.py`: prediction and output checks.
- `evaluate.py` and `diagnostics.json`: descriptive training-set evaluation and results.
- `train.py` and `new-training-split.json`: the separate optional training recipe.
- `weights/`: both final model files and their source manifest.
- `requirements.txt`, `architecture.json`, and `provenance.json`: environment and model details.
- `NOTICE` and `LICENSE`: attribution and licensing information.

## Reproduce

1. Install Git LFS, then clone the complete repository and run `git lfs pull`.
2. Create a Python environment; the canonical environment and exact package versions
   are recorded in `provenance.json` and `requirements.txt`.
3. Run `pip install -r requirements.txt`.
4. Open `pipeline.ipynb`, set its `IMAGES` path to official `test/test_images`, and run all cells.
   Alternatively run `python inference.py --images /path/to/test/test_images --device mps`.
5. Read `work/submission.csv` and its validation/hash receipt.

The canonical replay used Apple MPS. CUDA and CPU are supported; numerical backend
differences can change threshold-edge predictions, so exact byte equality is only
claimed for the independently verified canonical environment. No filename-derived
time, site, neighbouring observations or metadata is used by the predictor.

## Method and result

YOLO26x-seg and YOLOv8x-seg, each at 2048 resolution, confidence 0.5, box NMS IoU 0,
maximum 100 detections. Pool instance masks; confidence-order mask NMS at IoU 0.5;
paint masks into disjoint native 2048x2048 regions; discard remnants below five pixels.
Keep available predictions when one model is empty. Submission 58351849 scored
**0.40 public PQ**, with 1,317 masks over 178 of 180 images. Images with zero masks
are processed but emit no CSV rows.


## Provenance and compliance

The two public model cards declare training on the 707 sanctioned competition
training images only. We verified exact public model versions, file hashes, model
cards, checkpoint training arguments and readable inference source. We did **not**
retrain or independently reconstruct their complete training histories. The weights
have seen all training observations, so training-set metrics are not presented as
independent validation of these models. Prediction uses only the supplied test image.
The report must preserve this limitation. Attribution and licences: `NOTICE`,
`LICENSE`, and `weights/manifest.json`.

Final eligibility also requires the official Google Form, its prediction CSV and PDF
uploads, and public repository access at the required time. The owner has not yet
approved publication; this package remains private for review.


## Complete scientific workflow

`pipeline.ipynb` includes default inference and validation, optional descriptive
training-set diagnostics through `evaluate.py`, and optional new scratch training
through `train.py`. `architecture.json` records the actual stored model structure.
`diagnostics.json` records fixed-set baseline metrics and observation bootstrap
intervals. The training exporter was checked on all 1,154 official annotator records:
566 training and 141 validation observations, with no observation shared across splits.
The new optional recipe is not a reconstruction of the upstream model training run.

The primary artifact source is this repository. Our owned alternative mirror is
https://www.kaggle.com/datasets/shepscientific/filament-segmentation-2026-review-assets .
The notebook never downloads an upstream model or another participant's dataset.
Original authorship remains credited in NOTICE and weights/manifest.json.

Resource guidance: the two stored files total 286,752,617 bytes. Inference holds two
large models and native masks. The local diagnostic timing was about two seconds per
image after initialization on Apple MPS; other hardware can differ. Training needs a
separate resource and storage budget and is disabled by default.

The notebook includes a runnable pixel-grid example of mask IoU, duplicate suppression, and confidence-ordered pixel assignment. The report explains the final method as a learning workflow.
