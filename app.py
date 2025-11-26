from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from flask_cors import CORS  # <-- allow cross-origin requests
from model import load_model, predict_image

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Max upload = 5MB
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024
ALLOWED_EXT = {'png', 'jpg', 'jpeg'}


# Load model once on startup
model = load_model()

@app.before_first_request
def warmup():
    print("Warming up model with dummy prediction...")
    try:
        from io import BytesIO
        from PIL import Image
        import numpy as np

        # Create a dummy image similar to your real input shape
        dummy_img = Image.fromarray(
            np.uint8(np.zeros((224, 224, 3)))  # adjust size if needed
        )

        # Call your existing prediction function
        _pred, _conf = predict_image(model, dummy_img)
        print("Warmup done.")
    except Exception as e:
        print("Warmup failed:", e)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT


@app.errorhandler(413)
def file_too_large(e):
    return jsonify({"success": False, "error": "Selected file too large (max 5MB)."}), 413


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Invalid file format, only accepts .png, .jpg, .jpeg"}), 400

    try:
        filename = secure_filename(file.filename)
        prediction, confidence = predict_image(model, file)
        return jsonify({
            "success": True,
            "prediction": prediction,
            "confidence": float(confidence)
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": "Prediction failed",
            "details": str(e)
        }), 500


@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
