import tensorflow as tf
import numpy as np
from PIL import Image

# Smaller input size to reduce computation
IMG_SIZE = 160  # MobileNetV2 supports 96, 128, 160, 192, 224


def load_model():
    """
    Load a smaller MobileNetV2 once for inference.
    alpha < 1.0 makes the network narrower (lighter).
    """
    model = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=True,
        alpha=0.35,                        # smaller, lighter model (0.35, 0.5, 0.75, 1.0)
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    model.trainable = False  # inference only
    return model


def preprocess(img: Image.Image) -> np.ndarray:
    """
    Resize and preprocess an RGB PIL image to MobileNetV2 format.
    """
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return np.expand_dims(img, axis=0)


def predict_image(model, file):
    """
    Reads a file-like object, runs through the model, and maps to CAT/DOG/OTHER.
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
