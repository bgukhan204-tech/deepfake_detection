import os
import gdown
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Ensure model directory exists
model_path = "model/deepfake_model.h5"

if not os.path.exists(model_path):
    print("Downloading model...")
    os.makedirs("model", exist_ok=True)
    url = "https://drive.google.com/uc?id=1PRQ2PNMJKJhJPYWkeAWpjksr6A26hSPF"
    gdown.download(url, model_path, quiet=False)

print("Loading model...")
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully.")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read the image file using OpenCV from the stream
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            return jsonify({'error': 'Invalid image format'}), 400

        # Preprocessing matching the standalone script
        img_array = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Face Detection
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        # Lower minNeighbors and scaleFactor to catch smaller/distant faces in camera photos
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(50, 50))
        
        face_detected = False
        if len(faces) > 0:
            # Find the largest face by area
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face
            
            # Increase padding to 30% to capture full head context instead of just facial features
            pad_x = int(w * 0.3)
            pad_y = int(h * 0.3)
            
            y1 = max(0, y - pad_y)
            y2 = min(img_array.shape[0], y + h + pad_y)
            x1 = max(0, x - pad_x)
            x2 = min(img_array.shape[1], x + w + pad_x)
            
            # Crop the face
            img = img_array[y1:y2, x1:x2]
            face_detected = True
        else:
            # Fallback to the whole image
            img = img_array

        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict
        prediction = model.predict(img)[0][0]
        confidence = float(prediction)

        if prediction > 0.5:
            # Prediction near 1.0 -> 'real' (index 1)
            return jsonify({
                'status': 'REAL',
                'confidence': confidence,
                'real_confidence': confidence,
                'fake_confidence': 1.0 - confidence,
                'message': '✅ REAL IMAGE',
                'face_detected': face_detected
            })
        else:
            # Prediction near 0.0 -> 'fake' (index 0)
            return jsonify({
                'status': 'MANIPULATED',
                'confidence': 1.0 - confidence,
                'real_confidence': confidence,
                'fake_confidence': 1.0 - confidence,
                'message': '❌ MANIPULATED IMAGE',
                'face_detected': face_detected
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
