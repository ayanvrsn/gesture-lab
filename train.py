"""Train frozen ImageNet MobileNetV3-Small + its two-layer head on gesture images."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_small,MobileNet_V3_Small_Weights
from PIL import Image
from prepare import ROOT,DATA,square,CLASSES

def tensor(im):
    x=np.asarray(square(im),dtype=np.float32)/255
    x=(x-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    return torch.from_numpy(x.transpose(2,0,1).copy())
class Dataset:
    def __init__(self,rows):self.rows=rows
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        with Image.open(DATA/r['file']) as im:return tensor(im),CLASSES.index(r['label'])
def scores(y,p):
    c=np.zeros((3,3),int);np.add.at(c,(y,p),1)
    return {'n':int(c.sum()),'accuracy':float(c.trace()/c.sum()),'macro_f1':float(np.mean(2*c.diagonal()/np.maximum(1,c.sum(0)+c.sum(1)))),'confusion_matrix':c.tolist()}
def main(weights):
    torch.set_num_threads(4);torch.manual_seed(17)
    if weights:
        net=mobilenet_v3_small();net.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
    else: net=mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    net.classifier[-1]=nn.Linear(1024,3)
    net.features.requires_grad_(False);net.eval()
    initial={k:v.clone() for k,v in net.classifier.state_dict().items()}
    rows=list(csv.DictReader((DATA/'manifest.csv').open()))
    X=[];Y=[]
    with torch.no_grad():
        for x,y in DataLoader(Dataset(rows),batch_size=48,num_workers=0):
            X.append(net.avgpool(net.features(x)).flatten(1));Y.append(y)
    X=torch.cat(X);Y=torch.cat(Y)
    def ids(split,orig=False):
        return torch.tensor([i for i,r in enumerate(rows) if r['split']==split and (not orig or r['kind']=='original')])
    out=ROOT/'results';out.mkdir(exist_ok=True)
    report={}
    for setting in ['original_only','augmented']:
        tr=ids('train',setting=='original_only');val=ids('validation');test=ids('test')
        torch.manual_seed(17);net.classifier.load_state_dict(initial)
        opt=torch.optim.AdamW(net.classifier.parameters(),lr=.0003)
        history=[];best=(-1.,-float('inf'));checkpoint=None
        for epoch in range(1,7):
            net.classifier.train()
            perm=torch.cat([tr[torch.randperm(len(tr))] for _ in range((5400+len(tr)-1)//len(tr))])[:5400]
            total=0.
            for ii in perm.split(96):
                opt.zero_grad();loss=nn.functional.cross_entropy(net.classifier(X[ii]),Y[ii]);loss.backward();opt.step();total+=float(loss.detach())*len(ii)
            net.classifier.eval()
            with torch.no_grad():
                logits=net.classifier(X[val]);v=scores(Y[val].numpy(),logits.argmax(1).numpy());vl=float(nn.functional.cross_entropy(logits,Y[val]))
            key=(v['macro_f1'],-vl)
            if key>best:
                best=key;bestepoch=epoch;checkpoint={k:v.detach().clone() for k,v in net.classifier.state_dict().items()}
            history.append([epoch,total/5400,vl,v['accuracy'],v['macro_f1']])
        net.classifier.load_state_dict(checkpoint);net.eval()
        with torch.no_grad(): pred=net.classifier(X[test]).argmax(1)
        report[setting]={'best_epoch':bestepoch,'validation_macro_f1':best[0],'test':scores(Y[test].numpy(),pred.numpy())}
        with (out/(setting+'_curves.csv')).open('w') as f:
            w=csv.writer(f);w.writerow(['epoch','train_loss','val_loss','val_accuracy','val_macro_f1']);w.writerows(history)
        with (out/(setting+'_predictions.csv')).open('w') as f:
            w=csv.writer(f);w.writerow(['path','true','predicted']);w.writerows((rows[i]['parent_path'],rows[i]['label'],CLASSES[p]) for i,p in zip(test,pred))
        if setting=='augmented':torch.save(net.state_dict(),out/'mobilenet_v3_small_gestures.pth')
        print(setting,report[setting],flush=True)
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--weights',help='Optional local torchvision ImageNet V1 checkpoint');main(p.parse_args().weights)
