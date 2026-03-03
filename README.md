# Deepfake Detection Project

This project detects manipulated (deepfake) images using deep learning.

## Features
- CNN with transfer learning (Xception)
- Real vs Fake classification
- Premium Web interface using Flask, HTML, CSS, and JS

## How to Run
1. Navigate to the project directory:
   ```bash
   cd deepfake_detection
   ```
2. Activate your python virtual environment. (If you don't have one, create it and run `pip install -r requirements.txt`). Otherwise:
   ```bash
   ..\venv\Scripts\activate
   ```
3. Run the Flask Web Application:
   ```bash
   python app_web.py
   ```
4. Open your browser and navigate to `http://localhost:5000`

## Technologies
Python, TensorFlow, OpenCV, Flask, HTML/CSS/JS
