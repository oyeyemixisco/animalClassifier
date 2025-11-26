import tensorflow as tf
import numpy as np
from PIL import Image

# You can keep 224 if you like; just ensure app.py warmup uses same size
IMG_SIZE = 224

def load_model():
    """
    Load MobileNetV2 once for inference.
    We freeze it so TensorFlow doesn't keep gradient-related stuff.
    Optionally, use a smaller alpha to reduce memory usage.
    """
    model = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=True,
        alpha=0.75  # <--- optional: smaller than 1.0 to reduce size
    )
    model.trainable = False  # we are only inferring, never training
    return model


def preprocess(img: Image.Image) -> np.ndarray:
    """
    Resize and preprocess an RGB PIL image to the format expected by MobileNetV2.
    """
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return np.expand_dims(img, axis=0)


def predict_image(model, file):
    """
    Take a file-like object (e.g. uploaded file or BytesIO), run it through the model,
    and map Imagenet labels to CAT/DOG/OTHER.
    """
    try:
        img = Image.open(file).convert("RGB")
    except Exception:
        raise ValueError("Corrupted or unreadable image")

    x = preprocess(img)
    preds = model.predict(x)

    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0][0]
    class_name = decoded[1].lower()   # e.g. "labrador_retriever"
    confidence = float(decoded[2])

    dog_keywords = [
        "dog", "retriever", "pug", "husky", "shepherd", "terrier",
        "chihuahua", "maltese", "bulldog", "dalmatian", "great_dane",
        "rottweiler", "doberman", "shih-tzu", "collie", "beagle",
        "boxer", "whippet"
    ]

    cat_keywords = [
        "cat", "tabby", "tiger_cat", "siamese", "persian", "lynx"
    ]

    if any(k in class_name for k in cat_keywords):
        return "CAT", confidence

    if any(k in class_name for k in dog_keywords):
        return "DOG", confidence

    return "NOT A DOG OR CAT", confidence
