"""Usage: python prepare.py /path/to/CV_Dataset1.zip (new output folder only)."""
import csv, hashlib, io, json, sys, zipfile
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'
SIZE=160
SEED=2402
CLASSES=('peace','stop','thumbs_up')
def square(im):
    return ImageOps.pad(ImageOps.exif_transpose(im).convert('RGB'),(SIZE,SIZE),method=Image.Resampling.BILINEAR,color=(124,116,104))
def digest(b): return hashlib.sha256(b).hexdigest()
def prepare(src):
    rng=np.random.default_rng(SEED)
    if DATA.exists(): raise SystemExit('Start in a clean folder; data already exists.')
    rows=[]; parents=defaultdict(list); bygroup={}; seen=set()
    with zipfile.ZipFile(src) as z:
        assignments=list(csv.DictReader((ROOT/'source_assignments.csv').open()))
        incoming={n for n in z.namelist() if not n.startswith('__MACOSX/') and n.lower().endswith('.jpg')}
        assert incoming=={r['path'] for r in assignments}
        for r in assignments:
            path,label,group,split=[r[k] for k in ('path','label','scene','split')]
            b=z.read(path); source=digest(b)
            assert source==r['sha256']
            if split=='excluded': continue
            assert split in ('train','validation','test') and label in CLASSES
            assert bygroup.setdefault(group,split)==split
            assert source not in seen; seen.add(source)
            name=f'{split}/{label}/{source[:12]}_original.jpg'
            dest=DATA/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
            rows.append([name,split,label,group,path,source,'original',0,0,1,1,0,0,''])
            if split=='train': parents[label].append((square(Image.open(io.BytesIO(b))),group,path,source))
    for label in CLASSES:
        items=parents[label]
        for j in range(1800-len(items)):
            im,group,path,source=items[j%len(items)]
            seed=int(rng.integers(2**31)); r=np.random.default_rng(seed)
            deg=float(r.uniform(-15,15));flip=int(r.random()<.5)
            bright,contrast=map(float,r.uniform(.75,1.25,2))
            sigma=float(r.uniform(0.5,5)) if r.random()<.6 else 0
            blur=float(r.uniform(.25,1.0)) if r.random()<.4 else 0
            a=ImageOps.mirror(im) if flip else im
            a=a.rotate(deg,Image.Resampling.BILINEAR,fillcolor=(124,116,104))
            a=ImageEnhance.Contrast(ImageEnhance.Brightness(a).enhance(bright)).enhance(contrast)
            if blur: a=a.filter(ImageFilter.GaussianBlur(blur))
            if sigma: a=Image.fromarray(np.clip(np.asarray(a).astype(np.float32)+r.normal(0,sigma,(SIZE,SIZE,3)),0,255).astype(np.uint8))
            name=f'train/{label}/{source[:12]}_aug{j:04}.jpg'
            a.save(DATA/name,quality=82)
            rows.append([name,'train',label,group,path,source,'augmented',deg,flip,bright,contrast,sigma,blur,seed])
    with (DATA/'manifest.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow('file split label scene parent_path parent_sha256 kind rotation_deg mirror brightness contrast noise_sigma blur_radius seed'.split());w.writerows(rows)
    for split in ['train','validation','test']:
        part=[r for r in rows if r[1]==split];print(split,len(part),dict(Counter(r[2] for r in part)),dict(Counter(r[6] for r in part)))
    assert all(r[1]=='train' for r in rows if r[6]=='augmented')
    assert len(list(DATA.glob('train/*/*.jpg')))==5400
if __name__=='__main__': prepare(sys.argv[1])
