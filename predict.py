"""Offline inference; optionally write a Grad-CAM overlay for the predicted class."""
import argparse,time
import numpy as np
import torch
from torchvision.models import mobilenet_v3_small
from torch import nn
from PIL import Image
from prepare import ROOT,CLASSES,square
from train import tensor

def predict(path, overlay=None):
    torch.set_num_threads(4)
    net=mobilenet_v3_small();net.classifier[-1]=nn.Linear(1024,3)
    net.load_state_dict(torch.load(ROOT/'results/mobilenet_v3_small_gestures.pth',map_location='cpu',weights_only=True));net.eval()
    with Image.open(path) as image: inp=square(image);x=tensor(inp)[None]
    feat=net.features(x);score=net.classifier(net.avgpool(feat).flatten(1));i=int(score.argmax(1))
    if overlay:
        g=torch.autograd.grad(score[0,i],feat)[0];cam=(g.mean((2,3),keepdim=True)*feat).sum(1).clamp(min=0)[0].detach().numpy()
        cam=cam/max(1e-8,cam.max())
        mask=np.asarray(Image.fromarray(cam).resize(inp.size,Image.Resampling.BILINEAR)).clip(0,1)[...,None]*.65
        Image.fromarray((np.array(inp)*(1-mask)+np.array([255,65,0])*mask).astype('uint8')).save(overlay)
    print(CLASSES[i],dict(zip(CLASSES,score.detach().softmax(1)[0].tolist())))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('image');p.add_argument('--overlay');a=p.parse_args();predict(a.image,a.overlay)
