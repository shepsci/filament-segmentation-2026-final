# Solar Filament Segmentation 2026 - shepsci

PRIVATE REVIEW COPY. Publication requires completed owner review and explicit approval. Do not automatically publish at the deadline.

Minimal reproducible final-solution release. Model weights are **stored in this Git
repository using Git LFS**. The notebook SHA256-verifies the original PyTorch checkpoints.
No private dependencies, external model source, cached answers or retraining is needed.
Supply the official competition test images under their original licence.

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
are processed but emit no CSV rows. Use `--original-skip` to reproduce the 1,309-mask
upstream control (58351496, also 0.40).


## Provenance and compliance

The two public model cards declare training on the 707 sanctioned competition
training images only. We verified exact public model versions, file hashes, model
cards, checkpoint training arguments and readable inference source. We did **not**
retrain or independently reconstruct their complete training histories. The weights
have seen all training observations, so training-set metrics are not presented as
honest validation of these public models. No public MAGFiLO test annotations,
unverified Tushhaaarrrr assets, reference submission or encoded answers are used.
The report must preserve this limitation. Attribution and licences: `NOTICE`,
`LICENSE`, and `weights/manifest.json`.

The final four-page report and form confirmation are separate required deliverables;
this repository alone does not establish final eligibility.


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
