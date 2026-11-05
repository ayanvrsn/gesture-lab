"""Local web demo. Uploaded images stay in memory; no cloud API is used."""
import io, json, platform
from flask import Flask, request, jsonify, render_template
from PIL import Image, UnidentifiedImageError
from inference import Predictor, stress
from prepare import ROOT
app=Flask(__name__)
app.config['MAX_CONTENT_LENGTH']=10*1024*1024
Image.MAX_IMAGE_PIXELS=20000000
model=Predictor()
@app.get('/')
def index(): return render_template('index.html')
@app.get('/api/info')
def info():
    return jsonify(device='CPU',platform=platform.platform(),model='MobileNetV3-Small',input_size='160 × 160',metrics=json.loads((ROOT/'results/metrics.json').read_text()))
@app.post('/api/predict')
def predict():
    if 'image' not in request.files: return jsonify(error='Choose an image or capture a camera frame.'),400
    try:
        with Image.open(io.BytesIO(request.files['image'].read())) as source:
            source.load(); im=source.copy()
        kind=request.form.get('transform','original')
        explain=request.form.get('explain','1') == '1'
        include_images=request.form.get('include_images','1') == '1'
        result=model.run(stress(im,kind),explain=explain,include_images=include_images)
        result['transform']=kind
        return jsonify(result)
    except (UnidentifiedImageError,OSError,ValueError,Image.DecompressionBombError):
        return jsonify(error='Invalid image, unsupported transformation, or image too large.'),400
@app.errorhandler(413)
def too_large(error): return jsonify(error='Image must be smaller than 10 MB.'),413
if __name__=='__main__': app.run(host='127.0.0.1',port=8000,debug=False)
