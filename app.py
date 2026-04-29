# app.py

```python
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

app = Flask(__name__)

# -----------------------------
# Configuration
# -----------------------------
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'model/skin_cancer_model.h5'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if missing
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------
# Load AI Model
# -----------------------------
model = load_model(MODEL_PATH)

# Example classes
class_names = [
    'Benign Lesion',
    'Melanoma',
    'Basal Cell Carcinoma',
    'Squamous Cell Carcinoma'
]

# -----------------------------
# Helper Functions
# -----------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))

    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def predict_skin_cancer(img_path):
    processed_img = preprocess_image(img_path)

    prediction = model.predict(processed_img)

    predicted_class = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100

    result = class_names[predicted_class]

    # Risk logic
    if confidence >= 80:
        risk_level = 'High Risk'
    elif confidence >= 50:
        risk_level = 'Medium Risk'
    else:
        risk_level = 'Low Risk'

    return result, round(confidence, 2), risk_level

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/scan')
def scan():
    return render_template('scan.html')


@app.route('/analytics')
def analytics():
    return render_template('analytics.html')


@app.route('/patients')
def patients():
    return render_template('patients.html')


@app.route('/settings')
def settings():
    return render_template('settings.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Predict using AI model
        result, confidence, risk_level = predict_skin_cancer(filepath)

        return render_template(
            'result.html',
            filename=filename,
            prediction=result,
            confidence=confidence,
            risk_level=risk_level
        )

    return jsonify({'error': 'Invalid file type'})


@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result, confidence, risk_level = predict_skin_cancer(filepath)

        return jsonify({
            'prediction': result,
            'confidence': confidence,
            'risk_level': risk_level,
            'image_path': filepath
        })

    return jsonify({'error': 'Invalid file'})


@app.route('/result')
def result():
    return render_template('result.html')


@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500


# -----------------------------
# Run App
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)
```

---

# requirements.txt

```text
flask
tensorflow
keras
numpy
opencv-python
pillow
pandas
matplotlib
scikit-learn
gunicorn
python-dotenv
werkzeug
```
