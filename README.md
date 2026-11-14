# Solar Filament Segmentation 2026 - shepsci

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

Generic box NMS IoU 0.5 and fixed horizontal-flip union each scored 0.39 and were
rejected. The original independently developed U-Net/GBM pipeline scored 0.37.
These are displayed public scores; they are not held-out training validation.

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
