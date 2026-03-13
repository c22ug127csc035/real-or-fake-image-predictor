import os
import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template

try:
    import onnxruntime as ort
    has_ort = True
except ImportError:
    has_ort = False

app = Flask(__name__)

# Model configuration
MODEL_PATH = "ai_real_detector.onnx"
model_session = None

if has_ort and os.path.exists(MODEL_PATH):
    try:
        model_session = ort.InferenceSession(MODEL_PATH)
        print(f"Loaded model from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Warning: {MODEL_PATH} not found or onnxruntime not installed. Using mock predictions.")

def preprocess_image(image_bytes):
    """
    Standard preprocessing for image classification models.
    Adjust this based on actual model requirements (e.g., shape, normalization).
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        # Convert to RGB (in case of grayscale or RGBA)
        image = image.convert('RGB')
        # Resize to typical 224x224
        image = image.resize((224, 224))
        
        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Typical models expect NCHW format: (batch_size, channels, height, width)
        # PIL image is usually (height, width, channels)
        img_array = np.transpose(img_array, (2, 0, 1)) # HWC to CHW
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        return img_array
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
        
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
        
    try:
        image_bytes = file.read()
        
        if model_session is None:
            # Fallback mock response if model is not present
            import random
            import time
            time.sleep(1) # Simulate processing time
            fake_prob = random.uniform(0.1, 0.9)
            result = "Fake" if fake_prob > 0.5 else "Real"
            return jsonify({
                "prediction": result,
                "confidence": round(fake_prob if result == "Fake" else 1 - fake_prob, 4),
                "warning": f"Model file '{MODEL_PATH}' not found. Showing mock result."
            })
            
        input_data = preprocess_image(image_bytes)
        if input_data is None:
            return jsonify({"error": "Failed to process image"}), 400
            
        # Get input name for ONNX runtime
        input_name = model_session.get_inputs()[0].name
        
        # Run inference
        outputs = model_session.run(None, {input_name: input_data})
        
        # Assume output is a single probability of being "Fake" or similar
        # Adjust logic depending on actual model output format
        score = float(np.squeeze(outputs[0]))
        
        # Simple thresholding
        prediction = "Fake" if score > 0.5 else "Real"
        confidence = float(score if prediction == "Fake" else 1.0 - score)
        
        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence, 4)
        })
        
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
