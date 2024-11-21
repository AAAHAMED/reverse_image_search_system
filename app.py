from flask import Flask, jsonify, request
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for testing cross-origin requests

# Basic route to verify the app is running
@app.route("/")
def home():
    return "Reverse Image Search API is running!"

# Test endpoint to confirm API functionality
@app.route("/test", methods=["GET", "POST"])
def test():
    if request.method == "GET":
        return jsonify({"message": "GET request successful!"})
    elif request.method == "POST":
        # Echo back the data sent in the request
        data = request.json
        return jsonify({"message": "POST request successful!", "data": data})

# Start Flask app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
