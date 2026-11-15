"""Optional new training from scratch using official training annotations only.

This is a documented, reproducible training option, not a reconstruction of the
upstream weights' unavailable training run. It never runs in the default notebook.
"""
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


def prepare(official_train_directory, output):
    """Export COCO polygons. Keep every annotator of an observation in one split."""
    official=Path(official_train_directory).resolve()
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError('Choose a fresh output directory; existing data are never overwritten')
    document=json.loads((official/'MAGFiLO_1.0_Annotations_kaggle2026_train.json').read_text())
    annotations=defaultdict(list)
    for annotation in document['annotations']:
        annotations[str(annotation['image_id'])].append(annotation)
    observations=sorted({Path(i['file_name']).stem for i in document['images']},
                        key=lambda s:hashlib.sha256(('shepsci-training-split-v1:'+s).encode()).hexdigest())
    val=set(observations[:round(.2*len(observations))])
    manifest=[]
    for split in ('train','val'):
        (output/'images'/split).mkdir(parents=True)
        (output/'labels'/split).mkdir(parents=True)
    for item in document['images']:
        iid=str(item['id'])
        if Path(iid).name!=iid or '/' in iid or '\\' in iid:
            raise ValueError('Unsafe annotation record identifier')
        stem=Path(item['file_name']).stem
        split='val' if stem in val else 'train'
        image=(official/'train_images'/item['file_name']).resolve()
        if image.parent!=(official/'train_images').resolve():
            raise ValueError('Image must be in the official training directory')
        os.link(image,output/'images'/split/f'{iid}.jpeg')
        rows=[]
        for annotation in annotations[iid]:
            polygons=annotation['segmentation']
            if not isinstance(polygons,list) or len(polygons)!=1:
                raise ValueError('Exporter requires one polygon per annotated instance; no silent approximation')
            points=np.asarray(polygons[0],dtype=float).reshape(-1,2)
            if len(points)<3 or not np.isfinite(points).all():
                raise ValueError('Invalid polygon')
            points=points/np.array([item['width'],item['height']])
            if (points<0).any() or (points>1).any():
                raise ValueError('Polygon outside the image')
            # All supplied chirality categories map to one filament class.
            rows.append('0 '+' '.join(f'{v:.9f}' for v in points.reshape(-1)))
        (output/'labels'/split/f'{iid}.txt').write_text('\n'.join(rows)+'\n')
        manifest.append({'annotation_record':iid,'observation':stem,'split':split})
    configuration={'path':str(output),'train':'images/train','val':'images/val','names':{0:'filament'}}
    (output/'data.yaml').write_text(yaml.safe_dump(configuration,sort_keys=False))
    (output/'split.json').write_text(json.dumps({'seed_rule':'SHA256(shepsci-training-split-v1:stem)',
        'train_observations':len(observations)-len(val),'val_observations':len(val),
        'records':manifest,'scope':'New scratch training option, not the original public weights training split'},indent=2))
    return output/'data.yaml'


def train_from_scratch(data_yaml, architecture_index=0, device='0', epochs=100):
    """Create a fresh network from the recorded architecture; acquire no weights."""
    from ultralytics import YOLO
    root=Path(__file__).resolve().parent
    architecture=json.loads((root/'architecture.json').read_text())[architecture_index]['architecture_yaml']
    directory=Path(data_yaml).resolve().parent
    # The library infers scale from the filename and overwrites the YAML scale.
    configuration=directory/Path(architecture['yaml_file']).name
    configuration.write_text(yaml.safe_dump(architecture,sort_keys=False))
    model=YOLO(str(configuration),task='segment')
    return model.train(data=str(Path(data_yaml).resolve()),pretrained=False,
                       imgsz=2048,epochs=epochs,batch=1,device=device,
                       optimizer='AdamW',lr0=.001,lrf=.01,weight_decay=.0001,
                       seed=3407,deterministic=True,workers=0,
                       mosaic=0.0,mixup=0.0,copy_paste=0.0,hsv_h=0.0,hsv_s=0.0,
                       hsv_v=.1,degrees=10.0,translate=.1,scale=.1,shear=10.0,
                       fliplr=.5,flipud=.5,patience=25,project=str(directory/'runs'),name='scratch')
