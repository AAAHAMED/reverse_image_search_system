from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import requests

# Flask app initialization
app = Flask(__name__)
CORS(app)

# SAS URLs for train_features.npy and train_labels.npy
TRAIN_FEATURES_URL = "https://pythonmodule.blob.core.windows.net/module/train_features.npy?sp=r&st=2024-11-21T07:02:29Z&se=2024-11-23T15:02:29Z&spr=https&sv=2022-11-02&sr=c&sig=st3l%2F8qmaTbxFwSgOBFWMl6GaHSqIhNqbz6ENKt3m74%3D"
TRAIN_LABELS_URL = "https://pythonmodule.blob.core.windows.net/module/train_labels.npy?sp=r&st=2024-11-21T07:02:29Z&se=2024-11-23T15:02:29Z&spr=https&sv=2022-11-02&sr=c&sig=st3l%2F8qmaTbxFwSgOBFWMl6GaHSqIhNqbz6ENKt3m74%3D"

# Local file paths for downloaded blobs
TRAIN_FEATURES_FILE = "train_features.npy"
TRAIN_LABELS_FILE = "train_labels.npy"

# Pre-trained model setup (ResNet18)
model = models.resnet18(pretrained=True)
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])  # Remove classification layer
feature_extractor.eval()

# Transform for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to download files from Blob Storage
def download_blob(sas_url, local_path):
    response = requests.get(sas_url)
    if response.status_code == 200:
        with open(local_path, "wb") as file:
            file.write(response.content)
        print(f"Downloaded {local_path} successfully.")
    else:
        raise Exception(f"Failed to download {local_path}. Status code: {response.status_code}")

# Download dataset files
try:
    download_blob(TRAIN_FEATURES_URL, TRAIN_FEATURES_FILE)
    download_blob(TRAIN_LABELS_URL, TRAIN_LABELS_FILE)
except Exception as e:
    print(f"Error downloading files: {e}")

# Load dataset features and labels
train_features = np.load(TRAIN_FEATURES_FILE)
train_labels = np.load(TRAIN_LABELS_FILE)

@app.route("/")
def home():
    return "Reverse Image Search API is running!"

@app.route("/search", methods=["POST"])
def search():
    try:
        # Get the uploaded image
        file = request.files['image']
        img = Image.open(file).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)

        # Extract features for the query image
        with torch.no_grad():
            query_feature = feature_extractor(img_tensor).view(1, -1).numpy()

        # Compute cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_feature, train_features)
        top_k_indices = similarities.argsort()[0][-5:][::-1]

        # Prepare results (return labels and similarity scores)
        results = [
            {"label": int(train_labels[idx]), "score": float(similarities[0][idx])}
            for idx in top_k_indices
        ]
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
