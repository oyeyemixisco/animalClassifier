from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from model import load_model, predict_image

app = Flask(__name__)

# Max upload = 2MB
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024
ALLOWED_EXT = {'png', 'jpg', 'jpeg'}


# Load model once on startup
model = load_model()

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

        # Convert numeric confidence to High/Low
        confidence = float(confidence)
        if confidence >= 0.6:
            level = "High"
        else:
            level = "Low"
        
        return jsonify({
            "success": True,
            "prediction": prediction,
            "confidence": confidence,
            "level": level
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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
