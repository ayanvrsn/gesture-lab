"""Shared CPU inference, Grad-CAM and controlled stress transformations."""
import base64, io, time, threading
import numpy as np
import torch
from torch import nn
from torchvision.models import mobilenet_v3_small
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from prepare import ROOT, CLASSES, square
from train import tensor

class Predictor:
    def __init__(self):
        torch.set_num_threads(4)
        self.net = mobilenet_v3_small(weights=None)
        self.net.classifier[-1] = nn.Linear(1024, 3)
        self.net.load_state_dict(torch.load(ROOT/'results/mobilenet_v3_small_gestures.pth', map_location='cpu', weights_only=True))
        self.net.eval()
        self.lock = threading.Lock()
        with torch.no_grad(): self.net(torch.zeros(1,3,160,160))
    def run(self, im, explain=True, include_images=True):
        im = square(im)
        x = tensor(im)[None]
        with self.lock:
            start = time.perf_counter()
            with torch.set_grad_enabled(explain):
                feat = self.net.features(x)
                logits = self.net.classifier(self.net.avgpool(feat).flatten(1))
                probs = logits.detach().softmax(1)[0].tolist()
                winner = int(logits.argmax(1))
                inference_ms = (time.perf_counter()-start)*1000
                overlay = im
                if explain:
                    grad = torch.autograd.grad(logits[0,winner],feat)[0]
                    cam = (grad.mean((2,3),keepdim=True)*feat).sum(1).relu()[0].detach().numpy()
                    cam /= max(float(cam.max()),1e-8)
                    mask = np.asarray(Image.fromarray(cam).resize(im.size,Image.Resampling.BILINEAR)).clip(0,1)[...,None]*.65
                    overlay = Image.fromarray((np.array(im)*(1-mask)+np.array([255,65,0])*mask).astype('uint8'))
            total_ms = (time.perf_counter()-start)*1000
        result = {'label':CLASSES[winner], 'probabilities':dict(zip(CLASSES,probs)), 'inference_ms':round(inference_ms,2), 'explanation_total_ms':round(total_ms,2), 'explained':explain}
        if include_images:
            result['input'] = encode(im)
            if explain: result['heatmap'] = encode(overlay)
        return result

def encode(im):
    b=io.BytesIO(); im.save(b,format='JPEG'); return 'data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode()

def stress(im, kind):
    im=square(im)
    if kind=='dark': return ImageEnhance.Brightness(im).enhance(.5)
    if kind=='bright': return ImageEnhance.Brightness(im).enhance(1.5)
    if kind=='blur': return im.filter(ImageFilter.GaussianBlur(2))
    if kind=='flip': return ImageOps.mirror(im)
    if kind=='rotate': return im.rotate(15,Image.Resampling.BILINEAR,fillcolor=(124,116,104))
    if kind=='noise': return Image.fromarray(np.clip(np.array(im).astype(float)+np.random.default_rng(42).normal(0,10,(160,160,3)),0,255).astype('uint8'))
    if kind!='original': raise ValueError('Unknown transformation')
    return im
