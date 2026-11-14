"""Image-only two-model segmentation; no labels, metadata features or reference answers."""
import argparse, csv, hashlib, json, os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'work/config'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'work/mpl'))
import cv2
import numpy as np
import torch
from pycocotools import mask as mu
from ultralytics import YOLO

def file_sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def restore_weights():
    result=[]
    for m in json.loads((ROOT/'weights/manifest.json').read_text()):
        target=ROOT/m['path']
        assert target.stat().st_size==m['bytes'] and file_sha(target)==m['sha256'], 'Run git lfs pull to materialize repository weights'
        result.append(target)
    return result

def rle(mask):
    return mu.encode(np.asfortranarray(mask.astype(np.uint8)))['counts'].decode()

def decode(s):
    return mu.decode({'size':[2048,2048],'counts':s.encode()}).astype(bool)

def paint_and_nms(candidates):
    # Preserve the upstream confidence ordering, mask IoU=0.5 NMS and pixel painting.
    chosen=[]
    for c in sorted(candidates,key=lambda x:-x['score']):
        m=decode(c['rle'])
        if any((m & p).sum()/max(1,(m | p).sum())>0.5 for p in chosen):
            continue
        chosen.append(m)
    claimed=np.zeros((2048,2048),bool)
    result=[]
    for m in chosen:
        clean=m & ~claimed
        claimed |= m
        if clean.sum()>=5:
            result.append(rle(clean))
    return result

def predict(model,path,iou=0.0,flip=False,device="mps"):
    image=cv2.imread(str(path))
    if flip: image=np.ascontiguousarray(image[:,::-1])
    result=model.predict(image,device=device,imgsz=2048,conf=0.5,iou=iou,max_det=100,verbose=False)[0]
    if result.masks is None: return []
    masks=result.masks.data.cpu().numpy()
    scores=result.boxes.conf.cpu().numpy()
    rows=[]
    for mask,score in zip(masks,scores):
        mask=cv2.resize(mask.astype(np.uint8),(2048,2048),interpolation=cv2.INTER_NEAREST)
        if flip: mask=np.ascontiguousarray(mask[:,::-1])
        rows.append({'rle':rle(mask),'score':float(score)})
    return rows

def write_submission(path,by_image):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['filament_id','segmentation_rle']); w.writeheader()
        for stem in sorted(by_image):
            for k,c in enumerate(by_image[stem],1):
                w.writerow({'filament_id':f'{stem}_{k}','segmentation_rle':c})

def main(images,output,device=None,original_skip=False,limit=None):
    paths=sorted(Path(images).glob('*.jpeg'))
    assert paths, 'Point images at official competition test/test_images'
    if limit is None: assert len(paths)==180
    else: paths=paths[:limit]
    expected={p.stem for p in paths}
    device=device or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    models=[YOLO(str(p)) for p in restore_weights()]
    predictions={}
    for k,p in enumerate(paths):
        candidates=[predict(m,p,device=device) for m in models]
        predictions[p.stem]=[] if original_skip and any(not c for c in candidates) else paint_and_nms(sum(candidates,[]))
        print(f'{k+1}/{len(paths)} {len(predictions[p.stem])} instances',flush=True)
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True); write_submission(output,predictions)
    from validate import check
    assert check(output)==0
    rows=list(csv.DictReader(output.open()))
    assert all(r['filament_id'].rsplit('_',1)[0] in expected for r in rows)
    assert all(decode(r['segmentation_rle']).any() for r in rows)
    receipt={'device':device,'images_processed':len(paths),'images_with_predictions':sum(bool(p) for p in predictions.values()),
             'rows':len(rows),'sha256':file_sha(output),'original_skip':original_skip,
             'method':'imgsz2048/conf0.5/boxIoU0/maxdet100/maskIoU0.5/native disjoint paint/minarea5'}
    output.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2)); print(json.dumps(receipt,indent=2))
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--images',type=Path,required=True); p.add_argument('--output',type=Path,default=ROOT/'work/submission.csv'); p.add_argument('--device'); p.add_argument('--original-skip',action='store_true'); p.add_argument('--limit',type=int); a=p.parse_args()
    main(a.images,a.output,a.device,a.original_skip,a.limit)
