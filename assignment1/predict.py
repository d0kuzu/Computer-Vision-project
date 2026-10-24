"""Inference for one tightly cropped, dark-on-light digit image."""
from pathlib import Path
import argparse, json
import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent

def load_model():
    saved = torch.load(ROOT/'results/cnn.pt',map_location='cpu',weights_only=True)
    model = nn.Sequential(
        nn.Conv2d(1,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
        nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
        nn.Conv2d(64,128,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
        nn.Flatten(),nn.Linear(128*4*4,128),nn.ReLU(),nn.Dropout(.4),nn.Linear(128,len(saved['classes'])))
    model.load_state_dict(saved['state_dict']); model.eval()
    torch.set_num_threads(2)
    return model,saved['classes']

def preprocess(image):
    image = ImageOps.exif_transpose(image)
    if image.mode in ('RGBA','LA') or 'transparency' in image.info:
        rgba=image.convert('RGBA'); bg=Image.new('RGBA',rgba.size,'white'); bg.alpha_composite(rgba); image=bg
    im=ImageOps.invert(image.convert('L'))
    im=ImageOps.contain(im,(24,24),method=Image.Resampling.LANCZOS)
    canvas=Image.new('L',(32,32),0)
    canvas.paste(im,((32-im.width)//2,(32-im.height)//2))
    return torch.from_numpy(np.array(canvas,dtype=np.float32)/255).unsqueeze(0).unsqueeze(0)

def predict_image(model,classes,image):
    x=preprocess(image)
    with torch.inference_mode(): probabilities=model(x).softmax(1)[0].numpy()
    order=np.argsort(probabilities)[::-1]
    return {'digit':classes[int(order[0])], 'scores':[{'digit':classes[int(i)],'score':float(probabilities[i])} for i in order],
            'note':'Softmax scores are not calibrated confidence. Model trained on MNIST digits only.'}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('image',type=Path); args=parser.parse_args()
    model,classes=load_model()
    with Image.open(args.image) as image: print(json.dumps(predict_image(model,classes,image),indent=2))
