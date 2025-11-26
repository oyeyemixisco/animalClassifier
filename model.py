import tensorflow as tf
import numpy as np
from PIL import Image

IMG_SIZE = 224

def load_model():
    # MobileNetV2 pretrained on ImageNet
    model = tf.keras.applications.MobileNetV2(weights="imagenet")
    return model


def preprocess(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return np.expand_dims(img, axis=0)


def predict_image(model, file):
    try:
        img = Image.open(file).convert("RGB")
    except:
        raise ValueError("Corrupted or unreadable image")

    x = preprocess(img)
    preds = model.predict(x)
    # converts the raw prediction vector into human-readable labels
    # top=1  get the most probable class
    # [0][0]  extract the first batch and first (top) prediction
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0][0]

 # 'n02123045' → ImageNet class ID .... 'tabby' → Class name  ... 0.85 Confidence score (probability)
    class_name = decoded[1].lower()  
    confidence = decoded[2]


    # Improved Mapping (Fine tunning to fit the ML model classes to Dog/Cat/Unknown)
    dog_keywords = [
        "dog", "retriever", "pug", "husky", "shepherd", "terrier",
        "chihuahua", "maltese", "bulldog", "dalmatian", "great_dane",
        "rottweiler", "doberman", "shih-tzu", "collie", "beagle",
        "boxer", "whippet"
    ]

    cat_keywords = [
        "cat", "tabby", "tiger_cat", "siamese", "persian", "lynx"
    ]

    # check if class name contains any of the known patterns
    if any(k in class_name for k in cat_keywords):
        return "CAT", confidence

    if any(k in class_name for k in dog_keywords):
        return "DOG", confidence

    # fallback for unknown animals
    return "Unknown (Not Dog nor Cat)", confidence

    try:
        img = Image.open(file).convert("RGB")
    except:
        raise ValueError("Corrupted or unreadable image")

    x = preprocess(img)

    preds = model.predict(x)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0][0]

    class_name = decoded[1].lower()   # label like "tabby", "persian_cat", "chihuahua"
    confidence = decoded[2]

    # Simple mapping: if label contains cat or dog family
    if "cat" in class_name:
        return "cat", confidence
    elif "dog" in class_name:
        return "dog", confidence
    else:
        return "unknown", confidence
