"""Offline adapter for reviewed Step-Video-T2V videos. Does not generate videos.
Input JSON list: {video, label, model, prompt, seed, reviewed: true}.
Outputs remain separate from train/validation/test to prevent leakage.
"""
import argparse,json,subprocess,tempfile,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from inference import Predictor
from prepare import CLASSES
from PIL import Image

def main(manifest,output):
    manifest=Path(manifest).resolve(); items=json.loads(manifest.read_text()); output=Path(output)
    if not items: raise ValueError('No reviewed clips supplied; no SOTA experiment can be claimed.')
    output.mkdir(parents=True,exist_ok=False); model=Predictor(); records=[]
    for i,r in enumerate(items):
        if r.get('reviewed') is not True or r.get('model')!='Step-Video-T2V' or r.get('label') not in CLASSES:
            raise ValueError('Every clip needs manual gesture review, a valid label and Step-Video-T2V provenance.')
        if not r.get('prompt') or 'seed' not in r: raise ValueError('Prompt and seed are required.')
        video=(manifest.parent/r['video']).resolve()
        clipdir=output/f'clip_{i:03d}';clipdir.mkdir()
        subprocess.run(['ffmpeg','-v','error','-i',str(video),'-vf','fps=1','-frames:v','30',str(clipdir/'%03d.jpg')],check=True)
        frames=sorted(clipdir.glob('*.jpg'))
        if not frames: raise ValueError('No frames decoded')
        predictions=[]
        for frame in frames:
            with Image.open(frame) as im: predictions.append(model.run(im,False)['label'])
        majority=max(CLASSES,key=predictions.count)
        records.append({**r,'sha256':hashlib.sha256(video.read_bytes()).hexdigest(),'synthetic':True,'predictions':predictions,'majority':majority,'frame_accuracy':predictions.count(r['label'])/len(predictions),'clip_correct':majority==r['label']})
    result={'status':'evaluated_user_supplied_synthetic_clips','clips':records,'clip_accuracy':sum(r['clip_correct'] for r in records)/len(records),'note':'Clip is the evaluation unit; adjacent frames are not independent. This is a synthetic stress test, not evidence of improved real-test accuracy.'}
    (output/'metrics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('manifest');p.add_argument('output');a=p.parse_args();main(a.manifest,a.output)
