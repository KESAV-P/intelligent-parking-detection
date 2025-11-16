import tensorflow as tf
import numpy as np
import cv2
import sys

MODEL_PATH = "slot_classifier.h5"

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = (128, 128)

def predict_slot(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print("Error: Cannot load image")
        return

    img_resized = cv2.resize(img, IMG_SIZE)
    img_norm = img_resized.astype("float32") / 255.0
    img_input = np.expand_dims(img_norm, axis=0)

    pred = model.predict(img_input)[0][0]

    if pred > 0.5:
        print("🚗 OCCUPIED (%.3f)" % pred)
    else:
        print("🟩 FREE (%.3f)" % pred)

    return pred


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_slot.py <image_path>")
        sys.exit(1)

    predict_slot(sys.argv[1])