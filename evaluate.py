"""Descriptive mask diagnostics on official labelled training images only.

These diagnostics are not held-out validation for the included public weights.
The PQ hit rule follows the host self-evaluation notebook. No test labels are read.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mu


def observation(name):
    return Path(str(name)).stem.split('-', 1)[-1]


def annotation_rle(seg, height=2048, width=2048):
    if isinstance(seg, dict):
        return mu.frPyObjects(seg, height, width) if isinstance(seg['counts'], list) else seg
    rles = mu.frPyObjects(seg, height, width)
    return mu.merge(rles) if isinstance(rles, list) else rles


def distribution(values):
    return {'n':len(values), 'mean':float(np.mean(values)),
            'p10':float(np.quantile(values,.1)), 'p50':float(np.quantile(values,.5)),
            'p90':float(np.quantile(values,.9))} if values else {'n':0}


def evaluate(prediction_csv, official_training_json, selected_stems):
    """Score each annotator record separately; include images with no predictions."""
    selected_stems = set(selected_stems)
    raw = json.loads(Path(official_training_json).read_text())
    metadata = {str(m['id']):m for m in raw['images']}
    allowed = {observation(m['file_name']) for m in metadata.values()}
    if not selected_stems <= allowed:
        raise ValueError('Evaluation accepts official training observations only')
    gt = defaultdict(list)
    for annotation in raw['annotations']:
        item = metadata[str(annotation['image_id'])]
        if observation(item['file_name']) in selected_stems:
            gt[str(annotation['image_id'])].append(annotation_rle(annotation['segmentation']))
    predictions = defaultdict(list)
    with Path(prediction_csv).open(newline='') as handle:
        for row in csv.DictReader(handle):
            stem = row['filament_id'].rsplit('_',1)[0]
            if stem not in selected_stems:
                raise ValueError('Prediction outside the declared evaluation set')
            predictions[stem].append({'size':[2048,2048], 'counts':row['segmentation_rle'].encode()})
    tp=fp=fn=fragmented=merged=0
    total_iou=0.0
    matched_ious=[]
    binary_ious=[]
    binary_dices=[]
    per_record=[]
    for iid, item in metadata.items():
        stem=observation(item['file_name'])
        if stem not in selected_stems:
            continue
        truth, pred = gt[iid], predictions[stem]
        matrix = np.asarray(mu.iou(pred,truth,[0]*len(truth))).reshape(len(pred),len(truth)) if pred and truth else np.zeros((len(pred),len(truth)))
        hit=matrix>.5
        n_tp=int(hit.sum())
        n_fp=int((~hit.any(axis=1)).sum())
        n_fn=int((~hit.any(axis=0)).sum())
        values=matrix[hit].tolist()
        tp+=n_tp; fp+=n_fp; fn+=n_fn; total_iou+=sum(values); matched_ious.extend(values)
        # Loose IoU > .1 is a documented diagnostic, not a hidden leaderboard rule.
        fragmented+=int(((matrix>.1).sum(axis=0)>1).sum())
        merged+=int(((matrix>.1).sum(axis=1)>1).sum())
        gt_union=mu.merge(truth) if truth else None
        pred_union=mu.merge(pred) if pred else None
        if gt_union and pred_union:
            v=float(mu.iou([pred_union],[gt_union],[0])[0,0])
        else:
            v=1.0 if not truth and not pred else 0.0
        binary_ious.append(v); binary_dices.append(2*v/(1+v))
        den=n_tp+.5*n_fp+.5*n_fn
        per_record.append({'annotation_record':iid,'observation':stem,'PQ':sum(values)/den if den else 0.0,
                           'TP':n_tp,'FP':n_fp,'FN':n_fn,'binary_IoU':v,'binary_Dice':2*v/(1+v)})
    denominator=tp+.5*fp+.5*fn
    return {'scope':'Descriptive training-set diagnostics; included weights have seen the training observations',
            'observations':len(selected_stems),'annotation_records':len(per_record),
            'PQ':total_iou/denominator if denominator else 0.0,'TP':tp,'FP':fp,'FN':fn,
            'matched_instance_IoU':distribution(matched_ious),
            'binary_IoU':distribution(binary_ious),'binary_Dice':distribution(binary_dices),
            'one_to_many_IoU_over_0_1':fragmented,'many_to_one_IoU_over_0_1':merged,
            'per_record':per_record}
