from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import io

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Load pre-trained model and feature data
model = models.resnet18(pretrained=True)
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])  # Remove classification layer
feature_extractor.eval()
train_features = np.load('train_features.npy')
train_labels = np.load('train_labels.npy')

# Image preprocessing function
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def home():
    return "Reverse Image Search API is running!"

# Endpoint to handle image upload and similarity search
@app.route('/search', methods=['POST'])
def search():
    try:
        # Get the uploaded image
        file = request.files['image']
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)  # Preprocess and add batch dimension

        # Extract features
        with torch.no_grad():
            query_feature = feature_extractor(img_tensor).view(1, -1).numpy()

        # Perform similarity search
        similarities = cosine_similarity(query_feature, train_features)
        top_k_indices = similarities.argsort()[0][-5:][::-1]  # Top 5 similar images

        # Prepare results (return indices and similarity scores as placeholders)
        results = [{'index': int(idx), 'score': float(similarities[0][idx])} for idx in top_k_indices]
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
