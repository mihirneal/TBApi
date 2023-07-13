from flask import Flask, request, jsonify, render_template
from PIL import Image
from torchvision import transforms
import torch
import io
import requests
import os
from io import BytesIO

app = Flask(__name__)

model = torch.load('model1-2.ckpt', map_location=torch.device('cpu'))
model.eval()

def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return my_transforms(image_bytes).unsqueeze(0)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    return y_hat

@app.route('/')
def home():
    return render_template('start.html')


@app.route('/predict')
def predict():
    img_url = request.args.get('url')
    img_data = requests.get(img_url).content
    img = Image.open(BytesIO(img_data))
    class_id = get_prediction(image_bytes=img)


    return render_template('index.html', res=class_id.item())



if __name__ == '__main__':
    app.run(host= '0.0.0.0', port=8095)