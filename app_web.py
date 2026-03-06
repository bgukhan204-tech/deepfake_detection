import os
import cv2
import numpy as np
import gc
from PIL import Image
from flask import Flask, request, jsonify, render_template

# Detect environment
try:
    # On most cloud servers we use tflite-runtime for speed/memory
    import tflite_runtime.interpreter as tflite
except ImportError:
    # Fallback for local testing where full tensorflow might be installed
    import tensorflow.lite as tflite

app = Flask(__name__)

# Load face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Global interpreter and details
interpreter = None
input_details = None
output_details = None

def load_model():
    global interpreter, input_details, output_details
    if interpreter is not None:
        return interpreter, input_details, output_details
        
    # We now expect the .tflite model in the model/ folder
    model_path = "model/deepfake_model.tflite"
    if not os.path.exists(model_path):
        # Fallback for root
        model_path = "deepfake_model.tflite"
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"TFLite model not found at {model_path}. Please run convert_to_tflite.py locally.")
        
    # Log the file size to verify it uploaded correctly to Render
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"🚀 Loading TFLite model from {model_path} (Size: {size_mb:.2f} MB)...")
    
    if size_mb < 0.1:
        print("❌ CRITICAL ERROR: The model file on the server is EMPTY or corrupted (size ~0MB).")
        raise ValueError("Model file is empty. Please re-upload to GitHub.")

    # USE MODEL_CONTENT instead of model_path to bypass 'mmap' issues on cloud servers
    with open(model_path, 'rb') as f:
        model_content = f.read()
        
    # Initialize interpreter from buffer
    interpreter = tflite.Interpreter(model_content=model_content)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("✅ TFLite model loaded successfully from buffer.")
    return interpreter, input_details, output_details

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return "OK", 200

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        print(f"DEBUG: Processing file {file.filename}")
        # Load TFLite interpreter
        interp, inp, out = load_model()
        
        # Read the image file using OpenCV
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            return jsonify({'error': 'Invalid image format'}), 400

        img_array = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Face Detection
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(50, 50))
        
        face_detected = False
        if len(faces) > 0:
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face
            
            # Padding
            pad_x = int(w * 0.3)
            pad_y = int(h * 0.3)
            
            y1 = max(0, y - pad_y)
            y2 = min(img_array.shape[0], y + h + pad_y)
            x1 = max(0, x - pad_x)
            x2 = min(img_array.shape[1], x + w + pad_x)
            
            img = img_array[y1:y2, x1:x2]
            face_detected = True
        else:
            img = img_array
        
        # Preprocessing
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0  # TFLite expects float32
        img = np.expand_dims(img, axis=0)

        # Run TFLite inference
        print("DEBUG: Running TFLite inference...")
        interp.set_tensor(inp[0]['index'], img)
        interp.invoke()
        prediction = interp.get_tensor(out[0]['index'])[0][0]
        
        confidence = float(prediction)
        print(f"DEBUG: Finished. Confidence: {confidence}")

        # Clean memory
        gc.collect()

        if prediction > 0.5:
            return jsonify({
                'status': 'REAL',
                'confidence': confidence,
                'real_confidence': confidence,
                'fake_confidence': 1.0 - confidence,
                'message': '✅ REAL IMAGE',
                'face_detected': face_detected
            })
        else:
            return jsonify({
                'status': 'MANIPULATED',
                'confidence': 1.0 - confidence,
                'real_confidence': confidence,
                'fake_confidence': 1.0 - confidence,
                'message': '❌ MANIPULATED IMAGE',
                'face_detected': face_detected
            })

    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"ERROR: {error_msg}")
        return jsonify({'error': f"Inference failed: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)

