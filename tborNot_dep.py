from flask import Flask, request, jsonify
from PIL import Image
from torchvision import transforms
import torch
import io
import requests
import os
from io import BytesIO

app = Flask(__name__)

model = torch.load('model1-2.ckpt', map_location=torch.device('cpu'))

# Set the model to evaluation mode
model.eval()

# Define the transformations for the input image
def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    return y_hat

@app.route('/predict', methods=['POST'])
def predict():
    img_url = request.args.get('url')
    if img_url:
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        img_bytes = img.read()
        # img_t = transform_image(img_bytes)
        class_id = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_id': class_id.item()})
    else:
        return jsonify({'error': 'No image url provided'})

    # return render_template('res.html', message=class_id.item())

if __name__ == '__main__':
    app.run(host= '0.0.0.0', port=5001)
