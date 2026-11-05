"""Evaluate fixed deployed checkpoint by scene and controlled condition; no tuning."""
import csv,json,platform,time
from pathlib import Path
import numpy as np
from PIL import Image
from inference import Predictor,stress
from prepare import ROOT,DATA,CLASSES
from train import scores
model=Predictor();rows=list(csv.DictReader((DATA/'manifest.csv').open()))
rows=[r for r in rows if r['kind']=='original']
out=[];groups={};times=[]
for r in rows:
    with Image.open(DATA/r['file']) as im:
        result=model.run(im,False)
        i=CLASSES.index(result['label']); y=CLASSES.index(r['label'])
        groups.setdefault((r['split'],r['scene']),[[],[]]);groups[(r['split'],r['scene'])][0].append(y);groups[(r['split'],r['scene'])][1].append(i)
        times.append(result['inference_ms'])
        if r['split']=='test':
            for kind in ['original','dark','bright','blur','noise','rotate','flip']:
                d=model.run(stress(im,kind),False)
                out.append(dict(path=r['file'],true=r['label'],condition=kind,predicted=d['label']))
report={'environment':platform.platform(),'device':'CPU','threads':4,'forward_ms':{'n':len(times),'mean':float(np.mean(times)),'median':float(np.median(times)),'p95':float(np.percentile(times,95))},'scenes':{},'stress':{}}
for (split,scene),(y,p) in groups.items():report['scenes'][split+'/'+scene]=scores(np.array(y),np.array(p))
for kind in sorted(set(r['condition'] for r in out)):
    rr=[r for r in out if r['condition']==kind];report['stress'][kind]=scores(np.array([CLASSES.index(r['true']) for r in rr]),np.array([CLASSES.index(r['predicted']) for r in rr]))
(ROOT/'results/demo_evaluation.json').write_text(json.dumps(report,indent=2))
with (ROOT/'results/stress_predictions.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
print(json.dumps(report,indent=2))
