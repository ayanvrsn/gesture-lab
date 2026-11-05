"""Verify split isolation, parent hashes and the generated JPEGs."""
import csv,hashlib,collections
from pathlib import Path
from PIL import Image
from prepare import DATA
rows=list(csv.DictReader((DATA/'manifest.csv').open()))
assert len({r['file'] for r in rows})==len(rows)
sources={};groups={};stats=collections.Counter(); hashes={}
for r in rows:
    f=DATA/r['file']; b=f.read_bytes();s=r['split'];p=r['parent_sha256']
    assert sources.setdefault(p,s)==s and groups.setdefault(r['scene'],s)==s
    assert r['kind']=='original' or s=='train'
    h=hashlib.sha256(b).hexdigest()
    assert h not in hashes, 'identical image files: '+f.name
    hashes[h]=s
    if r['kind']=='original':assert h==p
    with Image.open(f) as im: im.verify()
    stats[(s,r['kind'])]+=1
parents={r['parent_sha256'] for r in rows if r['kind']=='original'}
assert set(sources)==parents and len(parents)==146
assert stats=={('train','original'):88,('train','augmented'):5312,('validation','original'):29,('test','original'):29}
print(stats)
